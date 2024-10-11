// Copyright (c) joisino 2022
// Released under the MIT license
// See LICENSE-wordtour for the full license text
// Modified by ymgw55 in 2024

#include<vector>
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<cmath>

int n, D;

std::vector<std::vector<double>> vs;

// Modified by ymgw55 in 2024
// Change: Replaced Euclidean distance with cosine distance for similarity metric.
// Original Euclidean distance code is commented out for reference.

// double d(int i, int j){
//   double res = 0;
//   for(int k = 0; k < D; k++){
//     double r = vs[i][k] - vs[j][k];
//     res += r * r;
//   }
//   return sqrt(res);
// }

double d(int i, int j) {
    double dot_product = 0;
    double norm_i = 0;
    double norm_j = 0;
    for (int k = 0; k < D; k++) {
        dot_product += vs[i][k] * vs[j][k];
        norm_i += vs[i][k] * vs[i][k];
        norm_j += vs[j][k] * vs[j][k];
    }
    norm_i = sqrt(norm_i);
    norm_j = sqrt(norm_j);
    double cosine_similarity = dot_product / (norm_i * norm_j);
    return 1 - cosine_similarity;  // Cosine distance computation
}

double score(std::vector<int> tour){
  double res = 0;
  for(int i = 0; i < n-1; i++){
    res += d(tour[i], tour[i+1]);
  }
  return res;
}

int main(int argc, char **argv){

  if(argc != 3){
    std::cerr << "Usage: " + std::string(argv[0]) + " [embedding file path] [#words]" << std::endl;
    exit(1);
  }

  std::ifstream ifs(argv[1]);
  n = std::stoi(argv[2]);

  for(int i = 0; i < n; i++){
    std::string line;
    std::getline(ifs, line);
    std::stringstream ss(line);
    std::string word;
    ss >> word;
    std::vector<double> v(0);
    while(!ss.eof()){
      double buf;
      ss >> buf;
      v.push_back(buf);
    }
    D = v.size();
    vs.push_back(v);
  }

  std::cout << "NAME: wordtour" << std::endl;
  std::cout << "TYPE: TSP" << std::endl;
  std::cout << "DIMENSION: " << n << std::endl;
  std::cout << "EDGE_WEIGHT_TYPE: EXPLICIT" << std::endl;
  std::cout << "EDGE_WEIGHT_FORMAT: UPPER_ROW" << std::endl;
  std::cout << "EDGE_WEIGHT_SECTION" << std::endl;
  for(int i = 0; i < n; i++){
    for(int j = i + 1; j < n; j++){
      std::cout << int(d(i, j) * 1000) << " ";
    }
    std::cout << std::endl;
  }
  
  return 0;
}
	