#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>
#include<map>
#include "Tensor.hpp"
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
};


/*
    数据重组器类，用于数据重组。
*/
template<typename ImplType>
class DataReconstructor{
    private:
        //dacpp::Tensor<ImplType> myTensor;      // 数据
        DataInfo myDataInfo;                   // 数据信息 形状
        Dac_Ops ops;                           // 作用于数据的算子组
        std::vector<PosNumber> posNumberList;  // 数据索引与物理位置的映射   
        void GetPos(std::vector<int> &pos, Dac_Ops &ops, int now) {
            if (now == this->myDataInfo.dim) {
                GetPosNumber(pos, ops);
                return;
            }
            for (int i = 0; i < this->myDataInfo.dimLength[now]; i++) {
                pos.push_back(i);
                GetPos(pos, ops, now + 1);
                pos.pop_back();
            }
        }
        void GetPosNumber(std::vector<int> &pos, Dac_Ops &ops) {
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
        void WriteRes(int &cnt, ImplType* res, std::vector<int> pos, ImplType* myTensor) {
            int stride = 1;
            for(int i=0;i<this->myDataInfo.dim;i++) stride*=this->myDataInfo.dimLength[i];
            int idx = 0;
            for(int i=0;i<this->myDataInfo.dim;i++){
                stride/=this->myDataInfo.dimLength[i];
                idx+=pos[i]*stride;
            }
            
        }

        /*
            将更新结果写入 myTensor
        */
        void WriteData(int &cnt, ImplType* res, std::vector<int> pos, dacpp::Tensor<ImplType> &myTensor) {
            myTensor.reviseValue(res[cnt++],pos);
        }
        
    public:
        DataReconstructor(){

        }

        /*
            通过 数据形状，作用于数据的算子，初始化数据重组器
        */
        void init(DataInfo dataInfo, Dac_Ops ops){
            this->myDataInfo=dataInfo;
            this->ops=ops;
            std::vector<int> pos; // 存位置的中间变量
            GetPos(pos, ops, 0);
            
            std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return (a.number==b.number)?a.pos<b.pos:a.number<b.number;});
            
        }

        /*
            将重组结果写入res长向量。
        */
        void Reconstruct(ImplType* res, ImplType* myTensor, sycl::queue& q){
            // int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteRes(cnt, res,this->posNumberList[i].pos,myTensor);
            // }
            // sycl::range<3> local(1, 1, this->posNumberList.size());
            // sycl::range<3> global(1, 1, 1);
            // q.submit([&](handler &h) {
            //     h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            //         const auto item_id = item.get_local_id(2);
            //         res[item_id]=myTensor[idx];
            //     });
            // }).wait();
        }

        /*
            用重组结果更新原数据
        */
        void UpdateData(ImplType* res, dacpp::Tensor<ImplType> &myTensor){
            int cnt=0;
            for (int i=0; i<this->posNumberList.size(); i++) {
                this->WriteData(cnt,res,this->posNumberList[i].pos,myTensor);
            }
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

int main() {
    auto selector = gpu_selector_v;
    queue q(selector);

    DataInfo matA_dataInfo;
    matA_dataInfo.dim = 3;
    matA_dataInfo.dimLength = {4, 4};

    RegularSlice si = RegularSlice("si", 2, 2);
	si.SetSplitSize(2);
	RegularSlice sk = RegularSlice("sk", 2, 2);
	sk.SetSplitSize(2);

    Dac_Ops matA_ops;
    si.setDimId(0);
    matA_ops.push_back(si);
    sk.setDimId(1);
    matA_ops.push_back(sk);
    
    DataReconstructor<int> matA_tool;
    matA_tool.init(matA_dataInfo, matA_ops);

    std::vector<int> dataA{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    std::vector<int> shapeA{4,4};
    dacpp::Tensor<int> matA(dataA,shapeA);

    int *d_matA=malloc_device<int>(16,q);
    int *r_matA=malloc_device<int>(16,q);
    int *h_matA=malloc_host<int>(16,q);
    matA.tensor2Array(h_matA);
    for(int i=0;i<16;i++) std::cout<<h_matA[i]<<" ";
    std::cout<<"\n";
    q.memcpy(d_matA,h_matA,16*sizeof(int)).wait();
    q.memcpy(h_matA,d_matA,16*sizeof(int)).wait();
    for(int i=0;i<16;i++) std::cout<<h_matA[i]<<" ";
    std::cout<<"\n";

    matA_tool.Reconstruct(r_matA,d_matA);
    q.memcpy(h_matA,r_matA,16*sizeof(int)).wait();
    for(int i=0;i<16;i++) std::cout<<h_matA[i]<<" ";
    std::cout<<"\n";
}