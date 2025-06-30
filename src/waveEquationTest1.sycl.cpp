#include <iostream>
#include <vector>
#include <cmath>
#include "Slice.h"
#include "Tensor.hpp"

using dacpp::Tensor;
using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}
// 网格参数
const int NX = 8;    // x方向网格数量
const int NY = 8;    // y方向网格数量
const double Lx = 10.0f; // x方向长度
const double Ly = 10.0f; // y方向长度
const double c = 1.0f;   // 波速
const int TIME_STEPS = 1000; // 时间步数
// 网格步长
const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);

// CFL条件
// const double dt = 0.5 * std::fmin(dx, dy) / c; // 满足稳定性条件





#include <sycl/sycl.hpp>
#include "DataReconstructor.h"

using namespace sycl;

void waveEq(double* cur, double* prev, double* next) 
{
    double dt = 0.5 * std::fmin(dx, dy) / c;

    double u_xx = (cur[(1+1) * 3 + 1] - 2.0f * cur[1 * 3 + 1] + cur[(1-1) * 3 + 1])/ (dx * dx);
    double u_yy = (cur[1 * 3 + (1+1)] - 2.0f * cur[1 * 3 + 1] + cur[1 * 3 + (1-1)])/ (dy * dy);

    next[0]=2.0f*cur[1 * 3 + 1]-prev[0]+(c * c)*dt*dt*(u_xx+u_yy);
}


// 生成函数调用
void waveEqShell(const dacpp::Tensor<double> & matCur, const dacpp::Tensor<double> & matPrev, dacpp::Tensor<double> & matNext) { 
    // 数据信息初始化
    DataInfo info_matCur;
    info_matCur.dim = matCur.getDim();
    for(int i = 0; i < info_matCur.dim; i++) info_matCur.dimLength.push_back(matCur.getShape(i));

    DataInfo info_matPrev;
    info_matPrev.dim = matPrev.getDim();
    for(int i = 0; i < info_matPrev.dim; i++) info_matPrev.dimLength.push_back(matPrev.getShape(i));

    DataInfo info_matNext;
    info_matNext.dim = matNext.getDim();
    for(int i = 0; i < info_matNext.dim; i++) info_matNext.dimLength.push_back(matNext.getShape(i));
    
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    // 算子初始化
    
    // 规则分区算子初始化
    RegularSlice sp1 = RegularSlice("sp1", 3, 1);
    sp1.SetSplitSize(6);
    // 规则分区算子初始化
    RegularSlice sp2 = RegularSlice("sp2", 3, 1);
    sp2.SetSplitSize(6);
    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.SetSplitSize(6);
    // 降维算子初始化
    Index idx2 = Index("idx2");
    idx2.SetSplitSize(6);
    // 数据重组
    
    // 数据重组
    DataReconstructor<double> matCur_tool;
    double* r_matCur=(double*)malloc(sizeof(double)*324);
    
    // 数据算子组初始化
    Dac_Ops matCur_ops;
    sp1.setDimId(0);
    sp1.setSplitLength(54);
    matCur_ops.push_back(sp1);
    sp2.setDimId(1);
    sp2.setSplitLength(9);
    matCur_ops.push_back(sp2);
    
    matCur_tool.init(info_matCur,matCur_ops);
    matCur_tool.Reconstruct(r_matCur,matCur);

    // std::cout<<"重组后的 r_matCur:\n";
    // for(int i = 0; i < 36; i++) {
    //     for (size_t j = 0; j < 9; j++){
    //         std::cout<<r_matCur[i*9+j]<<" ";
    //     }
    //     std::cout<<"\n";
    // }

    // 数据重组
    DataReconstructor<double> matPrev_tool;
    double* r_matPrev=(double*)malloc(sizeof(double)*36);
    


    // 数据算子组初始化
    Dac_Ops matPrev_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(6);
    matPrev_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(1);
    matPrev_ops.push_back(idx2);
    matPrev_tool.init(info_matPrev,matPrev_ops);
    matPrev_tool.Reconstruct(r_matPrev,matPrev);
    // 数据重组
    DataReconstructor<double> matNext_tool;
    double* r_matNext=(double*)malloc(sizeof(double)*36);
    
    // 数据算子组初始化
    Dac_Ops matNext_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(6);
    matNext_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(1);
    matNext_ops.push_back(idx2);
    matNext_tool.init(info_matNext,matNext_ops);
    matNext_tool.Reconstruct(r_matNext,matNext);

    // 设备内存分配
    
    // 设备内存分配
    double *d_matCur=malloc_device<double>(324,q);
    // 设备内存分配
    double *d_matPrev=malloc_device<double>(36,q);
    // 设备内存分配
    double *d_matNext=malloc_device<double>(36,q);
    // 数据移动
    
    // 数据移动
    q.memcpy(d_matCur,r_matCur,324*sizeof(double)).wait();
    // 数据移动
    q.memcpy(d_matPrev,r_matPrev,36*sizeof(double)).wait();
    // 数据初始化
    q.memset(d_matNext,0,36*sizeof(double)).wait();  
    // 内核执行
    
    //工作项划分
    sycl::range<3> local(1, 1, 36);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto sp1=(item_id/6+(0)+6)%6;
            const auto idx1=(item_id/6+(0)+6)%6;
            const auto sp2=(item_id+(0)+6)%6;
            const auto idx2=(item_id+(0)+6)%6;
            // 嵌入计算
			

            waveEq(d_matCur+(sp1*54+sp2*9),d_matPrev+(idx1*6+idx2*1),d_matNext+(idx1*6+idx2*1));
        });
    }).wait();
    
    
    // 归约
    
    // 返回计算结果
    
    // 归并结果返回
    q.memcpy(r_matNext, d_matNext, 36*sizeof(double)).wait();

    std::cout<<"r_nxt cur:\n";
    for (int i = 0; i < 6; i++) {
        for(int j = 0; j <6; j++){
            std::cout<<r_matNext[(i)*6+(j)]<<" ";
        }
        std::cout<<"\n";
    }

    matNext_tool.UpdateData(r_matNext,matNext);
    // 内存释放
    
    sycl::free(d_matCur, q);
    sycl::free(d_matPrev, q);
    sycl::free(d_matNext, q);
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now(); // 开始时间测量
    
    // 初始化波场
    vector<double> u_prev(NX * NY, 0.0f); // 前一步
    vector<double> u_curr(NX * NY, 0.0f);  // 当前步
    vector<double> u_next(NX * NY, 0.0f);  // 当前步

    // 初始条件：例如一个高斯脉冲
    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 0.5f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            double x = i * dx;
            double y = j * dy;
            u_prev[i*NX+j] = std::exp(-((x - Lx/2)*(x - Lx/2) + (y - Ly/2)*(y - Ly/2)) / (2 * sigma * sigma));
            //u_prev[i*NX+j] = i*NX+j;
        }
    }



    std::vector<int> shape_u_curr = {8, 8};
    Tensor<double> u_curr_tensor(u_curr, shape_u_curr);
    std::vector<double> u_prev_middle_points;
    for (int i = 1; i <= 6; i++) {
        std::vector<double> row;
        for (int j = 1; j <= 6; j++) {  
            u_prev_middle_points.push_back(static_cast<double>(u_prev[i*NY+j]));  
        }
        
    }
    std::vector<int> shape2= {6, 6};
    Tensor<double> u_prev_middle_tensor(u_prev_middle_points, shape2);

    std::vector<double> u_next_middle_points;
    for (int i = 1; i <= 6; i++) {
        std::vector<double> row;
        for (int j = 1; j <= 6; j++) {  
            u_next_middle_points.push_back(static_cast<double>(u_next[i*NY+j]));  
        }
    }
    std::vector<int> shape1= {6, 6};
    Tensor<double> u_next_middle_tensor(u_next_middle_points, shape1);

    for(int i = 0;i < TIME_STEPS; i++) {
        std::cout<<i<<" step:\n";

        double* tmp = new double[64];
        std::cout<<"now cur:\n";
        u_curr_tensor.tensor2Array(tmp);
        for (int i = 0; i < 8; i++) {
            for(int j = 0; j <8; j++){
                std::cout<<tmp[(i)*8+(j)]<<" ";
            }
            std::cout<<"\n";
        }

        // u_curr_tensor.print();
        //u_prev_middle_tensor.print();
        waveEqShell(u_curr_tensor, u_prev_middle_tensor, u_next_middle_tensor);
        // u_next_middle_tensor.print();

        for (int i = 1; i <= 6; i++) {
            for(int j = 1; j <=6; j++){
                double* data = new double[1];
                u_curr_tensor[i][j].tensor2Array(data);
                u_prev_middle_tensor[i-1][j-1].array2Tensor(data);
            }
        }

        for (int i = 1; i <= 6; i++) {
            for(int j = 1; j <=6; j++){
                double* data = new double[1];
                u_next_middle_tensor[i-1][j-1].tensor2Array(data);
                u_curr_tensor[i][j].array2Tensor(data);
            }
        }
        

 
        // 处理边界条件（绝热边界：导数为零）


        for (int i = 0; i < NX; ++i) {
            double* data = new double{0};
            u_curr_tensor[i][0].array2Tensor(data);             
            u_curr_tensor[i][NY-1].array2Tensor(data);
        }
        for (int j = 0; j < NY; ++j) {
            double* data = new double{0};
            u_curr_tensor[0][j].array2Tensor(data);              // 顶部边界
            u_curr_tensor[NX - 1][j].array2Tensor(data);
             // 底部边界
        }
        
        double* tmp1 = new double[64];
        std::cout<<"nxt cur:\n";
        u_curr_tensor.tensor2Array(tmp1);
        for (int i = 0; i < 8; i++) {
            for(int j = 0; j <8; j++){
                std::cout<<tmp1[(i)*8+(j)]<<" ";
            }
            std::cout<<"\n";
        }
        
    }

    auto end_time = std::chrono::high_resolution_clock::now(); // 结束时间测量
    std::chrono::duration<double> duration = end_time - start_time; // 计算持续时间
    std::cout << "总执行时间: " << duration.count() << " 秒" << std::endl; // 输出执行时间
    return 0;
}
