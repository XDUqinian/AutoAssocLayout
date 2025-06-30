#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include"dacInfo.h"

Dac_Op::Dac_Op(){}

/*
	通过 算子名称，算子划分数，算子作用的维度 创建算子。
*/
Dac_Op::Dac_Op(std::string Name,int SplitSize,int dimId){
	this->name = Name;
	this->split_size = SplitSize;
	this->dimId = dimId;
}
/*
	设置 算子作用的维度。
*/
void Dac_Op::setDimId(int id){
	this->dimId = id;
}
/*
	设置 算子划分的每份长度。
*/
void Dac_Op::setSplitLength(int len){
	this->split_length = len;
}

/*
	设置 算子划分的划分数。
*/
void Dac_Op::SetSplitSize(int split_size) {
	this->split_size = split_size;
}

/*
	设置 算子对应索引的计算表达式。
*/
void Dac_Op::setExp(std::string expression){
	this->expression = expression;
}
/*
	得到 算子对应索引的计算表达式。
*/
std::string Dac_Op::getExp(){
	return this->expression;
}


Dac_Ops::Dac_Ops(){
	this->size = 0;
}
void Dac_Ops::push_back(Dac_Op x){
	this->DacOps.push_back(x);
	this->size++;
}

void Dac_Ops::pop_back() {
	this->DacOps.pop_back();
	this->size--;
}

void Dac_Ops::clear(){
	while(this->size>0) {
		this->pop_back();
	}
}
Dac_Op& Dac_Ops::operator[](int i){
	return this->DacOps[i];
}


DacData::DacData(){

}
/*
	通过 数据名称，数据维数，作用于各个维度的算子 创建数据。
*/
DacData::DacData(std::string Name, int dim, Dac_Ops ops){
	this->name = Name;
	this->dim=dim;

	// 初始化维上长度为0，且该维度不被降维
	// for(int i=0;i<this->dim;i++) this->DimLength.push_back(0);
	// for(int i=0;i<this->dim;i++) this->isIndex.push_back(false);

	this->ops = ops;
	// int len = ops.size;
	// this->split_size = 1;
	// for(int i=0;i<len;i++){
	// 	this->split_size*=ops[i].split_size;
	// }
	// this->split_length = ops[0].split_size*ops[0].split_length/this->split_size;

	// for(int i=0;i<this->ops.size;i++)
	// {
	// 	this->setIndex(ops[i].dimId);
	// }
}
/*
	设置 数据在某维度上的长度。
*/
void DacData::setDimLength(int dimId,int len){
	if(dimId<this->DimLength.size()) this->DimLength[dimId]=len;
}
/*
	得到 数据在某维度上的长度。
*/
int DacData::getDimlength(int dimId){
	if(dimId<this->DimLength.size()) return this->DimLength[dimId];
	else return -1;
}
/*
	设置 数据在某维度上被降维。
*/
// void DacData::setIndex(int dimId){
// 	if(dimId<this->isIndex.size()) this->isIndex[dimId]=true;
// }
// /*
// 	检查 数据在某维度上是否被降维。
// */
// bool DacData::checkIndex(int dimId){
// 	if(dimId<this->isIndex.size()) return this->isIndex[dimId];
// 	else return false;
// }


Args::Args(){
	this->size=0;
}
void Args::push_back(DacData x){
	this->args.push_back(x);
	this->size++;
}
DacData& Args::operator[](int i){
	return this->args[i];
}

RegularSlice::RegularSlice(std::string name, int size, int stride) {
	this->name = name;
	this->stride = stride;
	this->size = size;
}

Index::Index(std::string name) {
	this->name = name;
	this->stride = 1;
	this->size = 1;
}
