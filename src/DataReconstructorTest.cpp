#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>
#include<map>
#include "ReconTensor.h"
#include "dacInfo.h"
#include <sycl/sycl.hpp>

using namespace sycl;



struct Range
{
    int start;
    int end;
};

struct DataInfo
{
    int dim;
    std::vector<int> dimLength;
};

struct PosNumber
{
    std::vector<int> number;   // 数据索引
    std::vector<int> pos;      // 物理位置
    std::vector<Range> region; // 数据单元的区域
    void printPos() {
        for(int i=0;i<this->pos.size();i++){
            std::cout<<this->pos[i]<<" ";
        }
        std::cout<<std::endl;
    }
};

struct PosNumberBuffer
{
    sycl::buffer<int> number;   // 数据索引
    int number_size;
    sycl::buffer<int> pos;      // 物理位置
    int pos_size;
    sycl::buffer<int> region_start; // 数据单元的区域
    sycl::buffer<int> region_end;
    int region_size
};


/*
    数据重组器类，用于数据重组。
*/
template<typename ImplType>
class DataReconstructor{
    public:
        DataInfo myDataInfo;                   // 数据信息 形状
        Dac_Ops ops;                           // 作用于数据的算子组
        std::vector<PosNumber> posNumberList;  // 数据索引与物理位置的映射
        std::vector<int> myIdx;
        sycl::buffer<int> myIdxBuffer;         // 用于在SYCL中传递 myIdx
        /*
            递归得到物理位置 pos 。
        */
        void GetPos(std::vector< std::vector<int> > &total_pos, std::vector<int> &pos, int now) {
            if (now == this->myDataInfo.dim) {
                total_pos.push_back(pos);
                return;
            }
            for (int i = 0; i < this->myDataInfo.dimLength[now]; i++) {
                pos.push_back(i);
                GetPos(total_pos, pos, now + 1);
                pos.pop_back();
            }
        }

        /*
            递归得到所有数据索引
        */
        void GetNumber(std::vector< std::vector<int> > &total_number, std::vector<int> &number, int now) {
            if (now == this->ops.size) {
                total_number.push_back(number);
                return;
            }
            for(int i=0;i<this->ops[now].split_size;i++) {
                number.push_back(i);
                GetNumber(total_number, number, now + 1);
                number.pop_back();
            }
        }
        void GetPosNumber(std::vector<int> &pos) {
            int dimNum = this->myDataInfo.dim;
            std::vector<Range> region;
            for (int i = 0; i < dimNum; i++) {
                Range r;
                r.start = 0;
                r.end = this->myDataInfo.dimLength[i];
                region.push_back(r);
            }
            
            std::vector<int> number;   // 存数据索引的中间变量
            RecursiveTraversal(number, pos, region, 0);
        }

        void GetPosNumber2(std::vector<int> &number) {
            int dimNum = this->myDataInfo.dim;
            std::vector<Range> region;
            for (int i = 0; i < dimNum; i++) {
                Range r;
                r.start = 0;
                r.end = this->myDataInfo.dimLength[i];
                region.push_back(r);
            }
            for (int i=0;i<number.size();i++) {
                int dimId = this->ops[i].dimId;
                region[dimId].start = region[dimId].start + number[i] * this->ops[i].stride;
                region[dimId].end = region[dimId].start + this->ops[i].size;
            }
            std::vector<int> pos;
            for (int i=0;i<dimNum;i++) {
                pos.push_back(region[i].start);
            }
            while(true) {
                PosNumber posNum;
                posNum.number = number;
                posNum.pos = pos;
                posNum.region = region;
                // posNum.printPos();
                this->posNumberList.push_back(posNum);
                int now = dimNum-1;
                while(now>=0) {
                    pos[now]++;
                    if (pos[now]<region[now].end) break;
                    pos[now] = region[now].start;
                    now--;
                }
                if(now<0) break;
            }
        }
        /*
            递归得到物理位置 pos 在当前算子组作用下应该映射的数据索引，以及划分好的数据单元的数据区域。
        */
        void RecursiveTraversal(std::vector<int> &number,std::vector<int> &pos, std::vector<Range> &region, int now) {
            if (now == ops.size) {
                PosNumber posNum;
                posNum.number = number;
                posNum.pos = pos;
                posNum.region = region;
                this->posNumberList.push_back(posNum);
                return;
            }
            Dac_Op op = ops[now];
            int id = pos[op.dimId]-region[op.dimId].start;
            int l;
            if (id-op.size<0) l=0;
            else l = ((id-op.size+1)%op.stride==0) ? (id-op.size+1)/op.stride : ((id-op.size+1+op.stride)&(-op.stride))/op.stride;
            int r = (region[op.dimId].start+(id&(-op.stride))+op.size<region[op.dimId].end) ? (id&(-op.stride))/op.stride : (region[op.dimId].end-op.size-region[op.dimId].start)/op.stride;
            for(int i = l; i <= r; i++) {
                int now_start = region[op.dimId].start;
                int now_end = region[op.dimId].end;
                number.push_back(i);
                region[op.dimId].start = now_start + i * op.stride;
                region[op.dimId].end = now_start + i * op.stride +  op.size;
                RecursiveTraversal(number, pos, region, now + 1);
                region[op.dimId].end = now_end;
                region[op.dimId].start = now_start;
                number.pop_back();
            }
        }
        
        /*
            将特定位置元素写入res长向量。
        */
        void WriteRes(int &cnt, ImplType* res, std::vector<int> pos, const dacpp::TensorBase<ImplType> &myTensor) {
            res[cnt++]=myTensor.getElement(pos);
        }

        /*
            将更新结果写入 myTensor
        */
        void WriteData(int &cnt, ImplType* res, std::vector<int> pos, dacpp::TensorBase<ImplType> &myTensor) {
            myTensor.reviseValue(res[cnt++],pos);
        }

        DataReconstructor(): myIdxBuffer(sycl::range<1>(0)){

        }

        /*
            通过 数据形状，作用于数据的算子，初始化数据重组器
        */
        void init(DataInfo dataInfo, Dac_Ops ops){
            this->myDataInfo=dataInfo;
            this->ops=ops;
            // std::vector< std::vector<int> > total_pos; //保存所有的位置
            // std::vector<int> pos; // 存位置的中间变量
            // GetPos(total_pos, pos, 0);
            // for(int i=0;i<total_pos.size();i++) {
            //     GetPosNumber(total_pos[i]);
            // }
            std::vector< std::vector<int> > total_number; // 保存所有的数据索引
            std::vector<int> number;
            GetNumber(total_number, number, 0);
            for(int i=0;i<total_number.size();i++) {
                GetPosNumber2(total_number[i]);
            }
            // printf("%d\n", this->posNumberList.size());
            std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return (a.number==b.number)?a.pos<b.pos:a.number<b.number;});
            for(int i=0;i<this->posNumberList.size();i++) this->posNumberList[i].printPos();
            for(int i=0;i<this->posNumberList.size();i++) this->myIdx.push_back(0);
            for(int i=0;i<this->posNumberList.size();i++){
                int stride = 1;
                for(int j=0;j<this->myDataInfo.dim;j++) stride*=this->myDataInfo.dimLength[j];
                int idx = 0;
                for(int j=0;j<this->myDataInfo.dim;j++) {
                    stride/=this->myDataInfo.dimLength[j];
                    idx+=this->posNumberList[i].pos[j]*stride;
                }
                this->myIdx[i]=idx;
            }

            this->myIdxBuffer = sycl::buffer<int>(this->myIdx.data(), sycl::range<1>(this->myIdx.size()));
        }

        /*
            将重组结果写入res长向量。
        */
        void Reconstruct(ImplType* res, ImplType* myTensor, sycl::queue& q){
            int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteRes(cnt, res,this->posNumberList[i].pos,myTensor);
            // }
            sycl::range<3> local(1, 1, this->posNumberList.size());
            sycl::range<3> global(1, 1, 1);
            q.submit([&](handler &h) {
                auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_local_id(2);
                    res[item_id]=myTensor[myIdxAccessor[item_id]];
                });
            }).wait();
        }

        /*
            用重组结果更新原数据
        */
        void UpdateData(ImplType* res, ImplType* myTensor, sycl::queue& q){
            // int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteData(cnt,res,this->posNumberList[i].pos,myTensor);
            // }
            sycl::range<3> local(1, 1, this->posNumberList.size());
            sycl::range<3> global(1, 1, 1);
            q.submit([&](handler &h) {
                auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_local_id(2);
                    myTensor[myIdxAccessor[item_id]] = res[item_id];
                });
            }).wait();
        }

        /*
            增加一个算子
        */
        void push_back(Dac_Op op) {
            this->ops.push_back(op);
            this->posNumberList.clear();
            std::vector<int> pos; // 存位置的中间变量
            GetPos(pos, this->ops, 0);
            std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return (a.number==b.number)?a.pos<b.pos:a.number<b.number;});
        }
        /*
            减少一个算子
        */
       void pop_back() {
            this->ops.pop_back();
            this->posNumberList.clear();
            std::vector<int> pos; // 存位置的中间变量
            GetPos(pos, this->ops, 0);
            std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return (a.number==b.number)?a.pos<b.pos:a.number<b.number;});
       }
};


void debug(int num){
    std::cout<<"ok"<<num<<"\n";
}
int main(){
    auto selector = gpu_selector_v;
    queue q(selector);

    int* res=(int*)malloc(sizeof(int)*16);
    
    std::vector<int> data{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    // std::vector<int> shape{4,4};
    dacpp::Tensor<int,2> myTensor({4,4},data);
    myTensor.print();

    DataInfo info_myTensor;
    info_myTensor.dim = myTensor.getDim();
    for(int i = 0; i < info_myTensor.dim; i++) info_myTensor.dimLength.push_back(myTensor.getShape(i));


    RegularSlice si = RegularSlice("si", 2, 2);
	si.SetSplitSize(2);
	RegularSlice sj = RegularSlice("sj", 2, 2);
	sj.SetSplitSize(2);
    Dac_Ops data_ops;
    si.setDimId(0);
	//si.setSplitLength(8);
	data_ops.push_back(si);
    sj.setDimId(1);
    //sj.setSplitLength(4);
    data_ops.push_back(sj);
    DataReconstructor<int> tool;
    tool.init(info_myTensor,data_ops);

    int h_myTensor[16];
    myTensor.tensor2Array(h_myTensor);
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<h_myTensor[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;

    int *d_myTensor=malloc_device<int>(16,q);
    q.memcpy(d_myTensor, h_myTensor, 16*sizeof(int)).wait();
    int *r_myTensor=malloc_device<int>(16,q);
    tool.Reconstruct(r_myTensor, d_myTensor,q);

    q.memcpy(h_myTensor, r_myTensor, 16*sizeof(int));
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<h_myTensor[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;

    int *d_myTensor2=malloc_device<int>(16,q);
    q.memset(d_myTensor2, 0, 16*sizeof(int)).wait();

    q.memcpy(h_myTensor, d_myTensor2, 16*sizeof(int));
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<h_myTensor[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;

    tool.UpdateData(r_myTensor, d_myTensor2,q);
    q.memcpy(h_myTensor, d_myTensor2, 16*sizeof(int));
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<h_myTensor[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }
    return 0;
}