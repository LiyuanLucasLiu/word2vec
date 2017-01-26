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
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1200
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define FREE(x) if (x != NULL) {free(x);}
#define CHECKNULL(x) if (x == NULL) {printf("Memory allocation failed\n"); exit(1);}
#define NRAND next_random = next_random * (unsigned long long)25214903917 + 11;
#define BREAD(x,f) fread(&x, sizeof(float), 1, f);
#define SREAD(x,f) fscanf(f, "%f ", &x);

typedef float real;                    // Precision of float numbers

struct supervision {
  long long function_id;
  long long label;
};

struct training_ins {
  long long id;
  long long c_num;
  long long *cList;
  long long sup_num;
  struct supervision *supList;
};

char test_file[MAX_STRING], model_file[MAX_STRING], test_result[MAX_STRING];
// char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
// struct vocab_word *vocab;
// long long  *cCount;
int binary = 1, debug_mode = 2;
long long c_size = 0, c_length = 100, l_size = 1, l_length = 400, d_size, tot_c_count = 0; //NONE_idx,
// real lambda1 = 0.3, lambda2 = 0.3;
long long ins_num = 2111, ins_count_actual = 0;
// long long iters = 10;
// long print_every = 1000;
// real alpha = 0.025, starting_alpha, sample = 1e-4;

struct training_ins * data;
long long * predicted_label;
real *c, *l, *lb;
real *o;
// real *sigTable, *expTable;

clock_t start;

// Reads a single word from a file, assuming comma + space + tab + EOL to be word boundaries
// 0: EOF, 1: comma, 2: tab, 3: \n, 4: space
inline int ReadWord(long long *word, FILE *fin) {
  // putchar('c');
  char ch;
  // printf("%lld\n", *word);
  int sgn = 1;
  *word = 0;
  // printf("%lld\n", *word);
  while (!feof(fin)) {
    ch = fgetc(fin);
    // putchar(ch);
    switch (ch) {
      case ',':
        return 1;
      case '\t':
        return 2;
      case '\n':
        return 3;
      case ' ':
        return 4;
      case '-':
        sgn = sgn * -1;
        break;
      default:
        *word = *word * 10 + ch - '0';
    }// Truncate too long words
  }
  *word = *word * sgn;
  return 0;
}

void DestroyNet() {
  FREE(lb)
  FREE(l)
  FREE(c)
  FREE(o)
}

void LoadTestingData(){
  FILE *fin = fopen(test_file, "r");
  if (fin == NULL) {
    fprintf(stderr, "no such file: %s\n", test_file);
    exit(1);
  }
  printf("curInsCount: %lld\n", ins_num);
  long long curInsCount = ins_num, a, b;
  
  data = (struct training_ins *) calloc(ins_num, sizeof(struct training_ins));
  predicted_label = (long long *) calloc(ins_num, sizeof(long long));
  while(curInsCount--){
    // printf("curInsCount: %lld\n", curInsCount);
    data[curInsCount].id = 1;
    // printf("curInsCount: %lld\n", data[curInsCount].id);
    ReadWord(&data[curInsCount].id, fin);
    // putchar('a');
    ReadWord(&data[curInsCount].c_num, fin);
    ReadWord(&data[curInsCount].sup_num, fin);
    data[curInsCount].cList = (long long *) calloc(data[curInsCount].c_num, sizeof(long long));
    data[curInsCount].supList = (struct supervision *) calloc(data[curInsCount].sup_num, sizeof(struct supervision));
    // printf("%lld, %lld, %lld\n", data[curInsCount].id, data[curInsCount].c_num, data[curInsCount].sup_num);

    for (a = data[curInsCount].c_num; a; --a) {
      ReadWord(&b, fin);
      if (b > c_size) c_size = b;
      data[curInsCount].cList[a-1] = b;
    }
    for (a = data[curInsCount].sup_num; a; --a) {
      ReadWord(&b, fin);
      if (b > l_size) l_size = b;
      data[curInsCount].supList[a-1].label = b;
      ReadWord(&b, fin);
      if (b > d_size) d_size = b;
      data[curInsCount].supList[a-1].function_id = b;
    }
  }
  c_size++; d_size++; l_size++;
  if ((debug_mode > 1)) {
    printf("load Done\n");
    printf("c_size: %lld, d_size: %lld, l_size: %lld\n", c_size, d_size, l_size);
  }
}

void TestModel() {
  int i, j, a, b;
  long long l1;
  real f, g;
  real *cs = (real *) calloc(c_length, sizeof(real));
  real *z = (real *) calloc(l_length, sizeof(real));
  long long correct = 0;
  for (i = 0; i < ins_num; ++i){
    struct training_ins * cur_ins = data+ i;
    //calculate z;
    for (j = 0; j < c_length; ++j)
      cs[j] = 0;
    for (a = 0; a < cur_ins->c_num; ++a) {
      l1 = c_length * cur_ins->cList[a];
      for (j = 0; j < c_length; ++j) cs[j] += c[l1 + j];
    }
    for (j = 0; j < c_length; ++j) cs[j] /= cur_ins->c_num;
    for (a = 0; a < l_length; ++a){
      z[a] = 0;
      l1 = a * c_length;
      for (j = 0; j < c_length; ++j) z[a] += cs[j] * o[l1 + j];
    }

    b = -1; g = 0;
    for (j = 0; j < l_size; ++j) {
      f = lb[j];
      l1 = j * l_length;
      for (a = 0; a < l_length; ++a) f += z[a] * l[l1 + a];
      if (-1 == b || f > g){
        g = f;
        b = j;
      }
      printf("%d, %d, %f, %f, %f, %f\n", i, j, f, z[0], l[l1], lb[j]);
    }
    predicted_label[i] = b;
    correct += (b == cur_ins->supList[0].label);
  }
  printf("totally %lld instances, correct %lld instances, accuracy %f", ins_num, correct, (real) correct / ins_num * 100);
}

void SaveResult() {
  FILE *fout = fopen(test_result, "w");
  int i;
  for (i = 0; i < ins_num; ++i) 
    fprintf(fout, "%lld,%lld\n", data[i].supList[0].label, predicted_label[i]); 
  fclose(fout);
}
void ReadModel() {  
  FILE *fi = fopen(model_file, "rb");
  long long a, b;
  if (fi == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", model_file);
    exit(1);
  }
  fscanf(fi, "%lld %lld %lld %lld %lld\n", &c_size, &c_length, &l_size, &l_length, &d_size);//, NONE_idx);
  a = posix_memalign((void **)&c, 128, (long long)c_size * c_length * sizeof(real));
  CHECKNULL(c)
  a = posix_memalign((void **)&l, 128, (long long)l_size * l_length * sizeof(real));
  CHECKNULL(l)
  a = posix_memalign((void **)&o, 128, (long long)c_length * l_length * sizeof(real));
  CHECKNULL(o)
  a = posix_memalign((void **)&lb, 128, (long long)l_size * sizeof(real));
  CHECKNULL(lb)
  if (binary) {
    for (b = 0; b < c_size; ++b) {
      for (a = 0; a < c_length; ++a) BREAD(c[b * c_length + a], fi)
    }
    for (b = 0; b < l_size; ++b) BREAD(lb[b], fi)
    for (b = 0; b < l_size; ++b) {
      for (a = 0; a < l_length; ++a) BREAD(l[b * l_length + a], fi)
    }
    for (b = 0; b < l_length; ++b) {
      for (a = 0; a < c_length; ++a) BREAD(o[b * c_length + a], fi)
    }
  } else {
    for (b = 0; b < c_size; ++b) {
      for (a = 0; a < c_length; ++a) SREAD(c[b * c_length + a], fi)
    }
    for (b = 0; b < l_size; ++b) SREAD(lb[b], fi)
    for (b = 0; b < l_size; ++b) {
      for (a = 0; a < l_length; ++a) SREAD(l[b * l_length + a], fi)
    }
    for (b = 0; b < l_length; ++b) {
      for (a = 0; a < c_length; ++a) SREAD(o[b * c_length + a], fi)
    }
  }
  fclose(fi);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the model\n");
    printf("\t-cleng <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-lleng <int>\n");
    printf("\t\tSet size of label vectors; default is 400\n");
    printf("\t-reSample <int>\n");
    printf("\t\tSet max skip length between words; default is 10\n");
    printf("\t-lambda1 <float>\n");
    printf("\t\tthe value of lambda");
    printf("\t-lambda2 <float>\n");
    printf("\t\tthe value of lambda");
    printf("\t-sample <float>\n");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 10)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-infer_together <int>\n");
    printf("\t\tInfering the true label with all parts of obj func\n");
    printf("\t-instances <int>\n");
    printf("\t\tthe number of instances in training set\n");
    printf("\t-iter <file>\n");
    printf("\t\tnumber of iters; default is 10\n");
    printf("\t-alpha_update_every <file>\n");
    printf("\t\tprint every # of instances; default is 1000\n");
    // printf("\t-none_idx <file>\n");
    // printf("\t\tthe index of None Type\n");
    printf("\nExamples:\n");
    printf("./recol -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -output /shared/data/ll2/CoType/data/intermediate/KBP/default.model\n\n");//-none_idx 5 
    return 0;
  }
  if ((i = ArgPos((char *)"-test", argc, argv)) > 0) strcpy(test_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-model", argc, argv)) > 0) strcpy(model_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(test_result, argv[i + 1]);
  if ((i = ArgPos((char *)"-instances", argc, argv)) > 0) ins_num = atoi(argv[i + 1]);

  printf("Loading training file %s\n", test_file);
  LoadTestingData();
  printf("Loading Model: %s\n", model_file);
  ReadModel();
  printf("start Testing \n ");
  TestModel();
  printf("\nSaving to %s\n", test_result);
  SaveResult();
  printf("releasing memory");
  DestroyNet();
  return 0;
}
