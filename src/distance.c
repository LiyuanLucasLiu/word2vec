//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h> // mac os x


const long long max_size = 2000;         // max length of strings
// const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f;
  char file_name[max_size];
  long long length, l_size, c_size, d_size, a, b, l1, l2, whethercosine;
  if (argc < 3) {
    printf("Usage: ./distance <FILE> <cosine or not>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  whethercosine = (argv[2][0] == '1');

  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &length);
  fscanf(f, "%lld", &c_size);
  fscanf(f, "%lld", &l_size);
  fscanf(f, "%lld", &d_size);
  float *c, *l, *d;

  printf("%lld, %lld, %lld, %lld\n", length, c_size, l_size, d_size);
  c = (float *)malloc((long long)length * c_size * sizeof(float));
  l = (float *)malloc((long long)length * l_size * sizeof(float));
  d = (float *)malloc((long long)length * d_size * sizeof(float));


  long long input_num, best_idx; 
  float best_score, tmp_score;

  for (a = 0; a < c_size; ++a){
    l1 = a * length;
    tmp_score = 0;
    for (b = 0; b < length; ++b) {
      fscanf(f, "%f", &c[l1 + b]);
      if (whethercosine) tmp_score+= c[l1+b] * c[l1+b];
    }
    if (whethercosine) {
      tmp_score = sqrt(tmp_score);
      for (b = 0; b < length; ++b) 
        c[l1 + b] /= tmp_score;
    }
  }
  printf("c load done\n");

  for (a = 0; a < l_size; ++a){
    l1 = a * length;
    tmp_score = 0;
    for (b = 0; b < length; ++b) {
      fscanf(f, "%f", &l[l1 + b]);
      if (whethercosine) tmp_score += l[l1 + b] * l[l1 + b];
    }
    if (whethercosine) {
      tmp_score = sqrt(tmp_score);
      for (b = 0; b < length; ++b)
        l[l1 + b] /= tmp_score;
    }
  }
  printf("l load done\n");

  for (a = 0; a < d_size; ++a){
    l1 = a * length;
    tmp_score = 0;
    for (b = 0; b < length; ++b) {
      fscanf(f, "%f", &d[l1 + b]);
      if (whethercosine) tmp_score += d[l1 + b] * d[l1 + b];
    }
    if (whethercosine) {
      tmp_score = sqrt(tmp_score);
      for (b = 0; b < length; ++b)
        d[l1 + b] /= tmp_score;
    }
  }
  printf("load done\n");
  fclose(f);

  while (1) {
    scanf("%lld", &input_num);
    if (input_num < 0 || input_num >= l_size)
      break;

    l2 = input_num * length;

    best_score = -INFINITY;

    for (a = 0; a < c_size; ++a){
      tmp_score = 0;
      l1 = a * length;
      for (b = 0; b < length; ++b)
        tmp_score += c[l1 + b] * l[l2 + b];
      if (tmp_score > best_score) {
        best_score = tmp_score;
        best_idx = a;
      }
    }

    printf("best_feature: %lld\n", best_idx);
  
    best_score = -INFINITY;

    for (a = 0; a < d_size; ++a){
      tmp_score = 0;
      l1 = a * length;
      for (b = 0; b < length; ++b)
        tmp_score += d[l1 + b] * l[l2 + b];
      if (tmp_score > best_score) {
        best_score = tmp_score;
        best_idx = a;
      }
    }

    printf("best_lf: %lld\n", best_idx);
  
  }
  return 0;
}
