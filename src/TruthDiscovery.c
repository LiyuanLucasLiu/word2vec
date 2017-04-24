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
#define FREE(x) //if (x != NULL) {free(x);}
#define CHECKNULL(x) if (x == NULL) {printf("Memory allocation failed\n"); exit(1);}
#define NRAND next_random = next_random * (unsigned long long)25214903917 + 11;
#define BREAD(x,f) fread(&x, sizeof(float), 1, f);
#define BWRITE(x,f) fwrite(&x , sizeof(real), 1, f);
#define SWRITE(x,f) fprintf(f, "%lf ", x);
#ifdef DEBUG
#define DDMode(f) {f;} 
#else
#define DDMode(f)
#endif

#ifdef GRADCLIP
#define GCLIP(x) ((grad = x), (grad > cur_grad_clip ? cur_grad_clip : (grad < -cur_grad_clip ? -cur_grad_clip : grad)))
#else
#define GCLIP(x) (x)
#endif

#define MINIVALUE 0.00001

#ifdef DROPOUT
#define DROPOUTRATIO 100000
#endif
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

char input_model[MAX_STRING], test_file[MAX_STRING], val_file[MAX_STRING];
char output_file[MAX_STRING];
// char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
// struct vocab_word *vocab;
long long  *cCount;
int binary = 1, debug_mode = 2, reSample = 20, min_count = 5, num_threads = 1, min_reduce = 1, infer_together = 0, no_lb = 1, no_db = 1, ignore_none = 0, error_log = 0, normL = 0, special_none = 0, printVal = 0 ;//, future work!! new labelling function...
long long c_size = 0, c_length = 100, l_size = 1, l_length = 400, d_size, tot_c_count = 0, NONE_idx = 6;
real lambda1 = 1, lambda2 = 1, lambda3 = 0, lambda4 = 0, lambda5 = 0, lambda6 = 0;
long long ins_num = 225977, ins_count_actual = 0; //133955 for pure_train
long long test_ins_num = 1900, val_ins_num = 211;
long long iters = 10;
long print_every = 1000;
real alpha = 0.025, starting_alpha, sample = 1e-4;
real grad_clip = 5;
long long useEntropy=1;

struct training_ins * data, *test_ins, *val_ins;
real *c, *l, *d, *cneg, *db, *lb;
real *o;
real ph1, ph2;
real *sigTable, *expTable, *tanhTable;
clock_t start;
#ifdef DROPOUT 
real dropout;
#endif

int negative = 5;
const int table_size = 1e9;
long long *table;

inline void copyIns(struct training_ins *To, struct training_ins *From){
  To->id = From->id;
  To->c_num = From->c_num;
  To->cList = From->cList;
  To->sup_num = From->sup_num;
  To->supList = From->supList;
}

void InitUnigramTable() {
  long long a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (long long *)malloc(table_size * sizeof(long long));
  if (table == NULL) {
    fprintf(stderr, "cannot allocate memory for the table\n");
    exit(1);
  }
  for (a = 0; a < c_size; a++) train_words_pow += pow(cCount[a], power);
  i = 0;
  d1 = pow(cCount[a], power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      while (cCount[++i] == 0) continue;
      d1 += pow(cCount[i], power) / (real)train_words_pow;
    }
    if (i >= c_size) i = c_size - 1;
  }
}

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

void normalizeL() {
  real sum;
  long long l1, i, j;
  for (i = 0; i < l_size; ++i) {
    l1 = i * l_length; sum = 0;
    for (j = 0; j < l_length; ++j){
      sum += l[l1 + j] * l[l1 + j];
    }
    sum = sqrt(sum);
    sum = sum > 0 ? sum : 1;
    for (j = 0; j < l_length; ++j){
      l[l1 +j] /= sum;
    }
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a;
  for (a = 0; a < c_size; a++) if (cCount[a] < min_reduce) cCount[a] = 0; 
}

void InitNet() {
  long long a, b;
  a = posix_memalign((void **)&c, 128, (long long)c_size * c_length * sizeof(real));
  CHECKNULL(c)
  a = posix_memalign((void **)&cneg, 128, (long long)c_size * c_length * sizeof(real));
  CHECKNULL(cneg)
  a = posix_memalign((void **)&l, 128, (long long)l_size * l_length * sizeof(real));
  CHECKNULL(l)
  a = posix_memalign((void **)&d, 128, (long long)d_size * l_length * sizeof(real));
  CHECKNULL(d)
  a = posix_memalign((void **)&o, 128, (long long)c_length * l_length * sizeof(real));
  CHECKNULL(o)
  if (0 == no_lb) {
    a = posix_memalign((void **)&lb, 128, (long long)l_size * sizeof(real));
    CHECKNULL(lb)
    for (b = 0; b < l_size; ++b) lb[b] = 0;
  }
  if (0 == no_db) {
    a = posix_memalign((void **)&db, 128, (long long)d_size * sizeof(real));
    CHECKNULL(db)
    for (b = 0; b < d_size; ++b) db[b] = 0;
  }
  cCount = (long long *) calloc(c_size,  sizeof(long long));
  CHECKNULL(cCount)
  memset(cCount, 0, c_size);
  // ph1 = (real*) calloc(d_size, sizeof(real));
  // CHECKNULL(ph1)
  // ph2 = (real*) calloc(d_size, sizeof(real));
  // CHECKNULL(ph2)
  ph2 = 1.0/l_size;
  ph1 = 1 - ph2;
  
  for (b = 0; b < c_size; ++b) for (a = 0; a < c_length; ++a) {
    c[b * c_length + a] = (rand() / (real)RAND_MAX - 0.5) / c_length;
    cneg[b * c_length + a] = 0;
  }
  // printf("%lld\n", b* c_length + a);
  // getchar();
  // for (b = 0; b < d_size; ++b) ph1[b] = 0.6;
  // for (b = 0; b < d_size; ++b) ph2[b] = 0.4;
  for (b = 0; b < l_size; ++b) for (a = 0; a < l_length; ++a)
    l[b * l_length + a] = 0;//(rand() / (real)RAND_MAX - 0.5) / l_length;
  for (b = 0; b < d_size; ++b) for (a = 0; a < l_length; ++a)
    d[b * l_length + a] = 0;//(rand() / (real)RAND_MAX - 0.5) / l_length; 
  for (b = 0; b < l_length; ++b) for (a = 0; a < c_length; ++a)
    o[b * c_length + a] = (rand() / (real)RAND_MAX - 0.5) / c_length;
}

void DestroyNet() {
  FREE(lb)
  FREE(db)
  FREE(l)
  FREE(d)
  FREE(c)
  FREE(cneg)
  FREE(o)
  FREE(cCount)
}


real calculateEntropy(real *tmp_predict_scores){
  real f, g = -INFINITY, sum_softmax = 0;
  long long i;
  for (i = 0; i < l_size; ++i) if (i != NONE_idx){
    g = g > tmp_predict_scores[i] ? g : tmp_predict_scores[i];
  }
  for (i = 0; i < l_size; ++i) if (i != NONE_idx){
    f = tmp_predict_scores[i] - g;
    if (f < -MAX_EXP) tmp_predict_scores[i] = exp(f);
    else if (f > MAX_EXP) printf("error! softmax over 1!\n");
    else tmp_predict_scores[i] = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    sum_softmax += tmp_predict_scores[i];
  }
  f = 0;
  for (i = 0; i < l_size; ++i) if (i != NONE_idx){
    g = tmp_predict_scores[i] / sum_softmax;
    f -= g * log(g);
  }
  return f;
}

real calculateInnerProd(real *tmp_predict_scores){
  long long i;
  real g = -INFINITY;
  for (i = 0; i < l_size; ++i) if (i != NONE_idx){
    g = g > tmp_predict_scores[i] ? g : tmp_predict_scores[i];
  }
  return -g;
}

void EvaluateModel() {
  long long i, j, a, b;
  long long l1;
  real f, g;
  real *cs = (real *) calloc(c_length, sizeof(real));
  real *z = (real *) calloc(l_length, sizeof(real));
  if (0 != ignore_none) {
    long long correct = 0;
    long long act_ins_num = 0;
    for (i = 0; i < val_ins_num; ++i){
      struct training_ins * cur_ins = val_ins + i;
      //calculate z;
      if (cur_ins->supList[0].label == NONE_idx)
        continue;
      for (j = 0; j < c_length; ++j)
        cs[j] = 0;
      for (a = 0; a < cur_ins->c_num; ++a) {
        l1 = c_length * cur_ins->cList[a];
        for (j = 0; j < c_length; ++j) cs[j] += c[l1 + j];
      }
      for (j = 0; j < c_length; ++j) cs[j] /= cur_ins->c_num;
      for (a = 0; a < l_length; ++a){
        g = 0;
        l1 = a * c_length;
        for (j = 0; j < c_length; ++j) g += cs[j] * o[l1 + j];
#ifdef ACTIVE
        if (g < -MAX_EXP) z[a] = -1;
        else if (g > MAX_EXP) z[a] = 1;
        else z[a] = tanhTable[(int)((g + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
#else
        z[a] = g;
#endif
      }
      // for (a = 0; a < l_length; ++a)
      // l2 = i * l_size;
      b = -1; g = 0;
      for (j = 0; j < l_size; ++j) if (j != NONE_idx) {
        if (0 == no_lb) f = lb[j];
        else f = 0;
        l1 = j * l_length;
        for (a = 0; a < l_length; ++a) f += z[a] * l[l1 + a];
        if (-1 == b || f > g){
          g = f;
          b = j;
        }
        if (debug_mode > 2) printf("%f, ", f);
        // DDMode(printf("%d, %d, %lld, %f, %f, %f\n", i, j, l2 + j, f, z[0], l[l1]));
        // scores[l2 + j] = f;
      }
      // predicted_label[i] = b;
      if (debug_mode > 2) printf("%lld, %lld, %lld\n", i, cur_ins->supList[0].label, b);
      correct += (b == cur_ins->supList[0].label);
      ++act_ins_num;
    }
    if (0 == printVal) printf("%f, ", (real) correct / act_ins_num * 100);
    correct = 0;
    act_ins_num = 0;
    for (i = 0; i < test_ins_num; ++i){
      struct training_ins * cur_ins = test_ins + i;
      //calculate z;
      if (cur_ins->supList[0].label == NONE_idx)
        continue;
      for (j = 0; j < c_length; ++j)
        cs[j] = 0;
      for (a = 0; a < cur_ins->c_num; ++a) {
        l1 = c_length * cur_ins->cList[a];
        for (j = 0; j < c_length; ++j) cs[j] += c[l1 + j];
      }
      for (j = 0; j < c_length; ++j) cs[j] /= cur_ins->c_num;
      for (a = 0; a < l_length; ++a){
        g = 0;
        l1 = a * c_length;
        for (j = 0; j < c_length; ++j) g += cs[j] * o[l1 + j];
#ifdef ACTIVE
        if (g < -MAX_EXP) z[a] = -1;
        else if (g > MAX_EXP) z[a] = 1;
        else z[a] = tanhTable[(int)((g + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
#else
        z[a] = g;
#endif
      }
      // for (a = 0; a < l_length; ++a)
      // l2 = i * l_size;
      b = -1; g = 0;
      for (j = 0; j < l_size; ++j) if (j != NONE_idx) {
        if (0 == no_lb) f = lb[j];
        else f = 0;
        l1 = j * l_length;
        for (a = 0; a < l_length; ++a) f += z[a] * l[l1 + a];
        if (-1 == b || f > g){
          g = f;
          b = j;
        }
        if (debug_mode > 2) printf("%f, ", f);
        // DDMode(printf("%d, %d, %lld, %f, %f, %f\n", i, j, l2 + j, f, z[0], l[l1]));
        // scores[l2 + j] = f;
      }
      // predicted_label[i] = b;
      if (printVal) printf("%lld, %lld\n", cur_ins->supList[0].label, b);
      if (debug_mode > 2) printf("%lld, %lld, %lld\n", i, cur_ins->supList[0].label, b);
      correct += (b == cur_ins->supList[0].label);
      ++act_ins_num;
    }
    if (0 == printVal) printf("%f\n", (real) correct / act_ins_num * 100);
  } else {
    long long correct = 0;
    long long act_ins_num = 0, act_pred_num = 0;
    real *entropy_list = (real *) calloc(val_ins_num, sizeof(real));
    long long *label_list = (long long *) calloc(val_ins_num, sizeof(long long));
    real *predict_scores = (real *) calloc(l_size, sizeof(real));

    for (i = 0; i < val_ins_num; ++i){
      struct training_ins * cur_ins = val_ins + i;
      //calculate z;
      act_ins_num += (cur_ins->supList[0].label == NONE_idx ? 0 : 1);
      for (j = 0; j < c_length; ++j)
        cs[j] = 0;
      for (a = 0; a < cur_ins->c_num; ++a) {
        l1 = c_length * cur_ins->cList[a];
        for (j = 0; j < c_length; ++j) cs[j] += c[l1 + j];
      }
      for (j = 0; j < c_length; ++j) cs[j] /= cur_ins->c_num;
      for (a = 0; a < l_length; ++a){
        g = 0;
        l1 = a * c_length;
        for (j = 0; j < c_length; ++j) g += cs[j] * o[l1 + j];
#ifdef ACTIVE
        if (g < -MAX_EXP) z[a] = -1;
        else if (g > MAX_EXP) z[a] = 1;
        else z[a] = tanhTable[(int)((g + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
#else
        z[a] = g;
#endif
      }
      // l2 = i * l_size;
      b = -1; g = 0;
      for (j = 0; j < l_size; ++j) {
        if (0 == no_lb) f = lb[j];
        else f = 0;
        l1 = j * l_length;
        for (a = 0; a < l_length; ++a) f += z[a] * l[l1 + a];
        if (-1 == b || f > g){
          g = f;
          b = j;
        }
        predict_scores[j] = f;
        // DDMode(printf("%d, %d, %lld, %f, %f, %f\n", i, j, l2 + j, f, z[0], l[l1]));
        // scores[l2 + j] = f;
      }
      label_list[i] = (b==cur_ins->supList[0].label) && (NONE_idx != b);
      if (NONE_idx != b) {
        if (useEntropy) entropy_list[i] = calculateEntropy(predict_scores);
        else entropy_list[i] = calculateInnerProd(predict_scores);
      } else {
        entropy_list[i] = INFINITY;
      }
    }
    real min_entropy = INFINITY, max_entropy = -INFINITY, best_pre = -INFINITY, best_rec = -INFINITY, best_f1 = -INFINITY, best_threshold = 1;

    for (i = 0; i < val_ins_num; ++i) if (entropy_list[i] < INFINITY) {
      min_entropy = min_entropy < entropy_list[i] ? min_entropy : entropy_list[i];
      max_entropy = max_entropy > entropy_list[i] ? max_entropy : entropy_list[i]; 
    }
    max_entropy = (max_entropy - min_entropy)/100;
    for (a = 1; a < 100; ++a) {
      correct = 0;
      act_pred_num = 0;
      f = min_entropy + max_entropy * a;
      for (i = 0; i < val_ins_num; ++i){
        if (entropy_list[i] < f) {
          correct += label_list[i];
          ++act_pred_num;
        }
      }
      if ((real) 200 * correct / (act_pred_num + act_ins_num) > best_f1) {
        best_f1 = (real) 200 * correct / (act_pred_num + act_ins_num);
        best_pre = 100*(correct+MINIVALUE)/(act_pred_num+MINIVALUE);
        best_rec = 100*(correct+MINIVALUE)/(act_ins_num+MINIVALUE);
        best_threshold = f;
      }
    }

    if (0 == printVal) printf("%f,%f,%f,", best_pre, best_rec, best_f1);

    if (printVal) printf("\n");

    for (i = 0; i < test_ins_num; ++i){
      struct training_ins * cur_ins = test_ins + i;
      //calculate z;
      act_ins_num += (cur_ins->supList[0].label == NONE_idx ? 0 : 1);
      for (j = 0; j < c_length; ++j)
        cs[j] = 0;
      for (a = 0; a < cur_ins->c_num; ++a) {
        l1 = c_length * cur_ins->cList[a];
        for (j = 0; j < c_length; ++j) cs[j] += c[l1 + j];
      }
      for (j = 0; j < c_length; ++j) cs[j] /= cur_ins->c_num;
      for (a = 0; a < l_length; ++a){
        g = 0;
        l1 = a * c_length;
        for (j = 0; j < c_length; ++j) g += cs[j] * o[l1 + j];
#ifdef ACTIVE
        if (g < -MAX_EXP) z[a] = -1;
        else if (g > MAX_EXP) z[a] = 1;
        else z[a] = tanhTable[(int)((g + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
#else
        z[a] = g;
#endif
      }
      // l2 = i * l_size;
      b = -1; g = 0;
      for (j = 0; j < l_size; ++j) {
        if (0 == no_lb) f = lb[j];
        else f = 0;
        l1 = j * l_length;
        for (a = 0; a < l_length; ++a) f += z[a] * l[l1 + a];
        if (-1 == b || f > g){
          g = f;
          b = j;
        }
        predict_scores[j] = f;
        // DDMode(printf("%d, %d, %lld, %f, %f, %f\n", i, j, l2 + j, f, z[0], l[l1]));
        // scores[l2 + j] = f;
      }
      if(printVal) printf("%f, ", g);

      // predicted_label[i] = b;
      if (useEntropy) g = calculateEntropy(predict_scores);
      else g = calculateInnerProd(predict_scores);
    
      if (printVal) printf("%f, %f, %d, %lld, %lld\n", g, best_threshold, g < best_threshold, b, cur_ins->supList[0].label);
      if (g < best_threshold && NONE_idx != b) {
        correct += (b == cur_ins->supList[0].label);
        ++act_pred_num;
      }
    }
    if (0 == printVal) printf("%f,%f,%f\n", (real)100*(correct+MINIVALUE)/(act_pred_num+MINIVALUE), (real)100*(correct+MINIVALUE)/(act_ins_num+MINIVALUE), (real)200*correct/(act_ins_num+act_pred_num));
  }
  FREE(cs);
  FREE(z);
}

void LoadTestingData(){
  FILE *fin = fopen(test_file, "r");
  if (fin == NULL) {
    fprintf(stderr, "no such file: %s\n", test_file);
    exit(1);
  }
  if (debug_mode > 1) printf("curInsCount: %lld\n", test_ins_num);
  long long curInsCount = test_ins_num, a, b;
  
  test_ins = (struct training_ins *) calloc(test_ins_num, sizeof(struct training_ins));
  while(curInsCount--){
    // printf("curInsCount: %lld\n", curInsCount);
    test_ins[curInsCount].id = 1;
    // printf("curInsCount: %lld\n", test_ins[curInsCount].id);
    ReadWord(&test_ins[curInsCount].id, fin);
    // putchar('a');
    ReadWord(&test_ins[curInsCount].c_num, fin);
    ReadWord(&test_ins[curInsCount].sup_num, fin);
    test_ins[curInsCount].cList = (long long *) calloc(test_ins[curInsCount].c_num, sizeof(long long));
    test_ins[curInsCount].supList = (struct supervision *) calloc(test_ins[curInsCount].sup_num, sizeof(struct supervision));
    // printf("%lld, %lld, %lld\n", test_ins[curInsCount].id, test_ins[curInsCount].c_num, test_ins[curInsCount].sup_num);

    for (a = test_ins[curInsCount].c_num; a; --a) {
      ReadWord(&b, fin);
      test_ins[curInsCount].cList[a-1] = b;
      // printf("(%lld)", b);
    }
    // printf("\n");
    for (a = test_ins[curInsCount].sup_num; a; --a) {
      ReadWord(&b, fin);
      test_ins[curInsCount].supList[a-1].label = b;
      ReadWord(&b, fin);
      test_ins[curInsCount].supList[a-1].function_id = b;
      // printf("(%lld, %lld)", test_ins[curInsCount].supList[a-1].label, test_ins[curInsCount].supList[a-1].function_id);
    }
    // printf("\n");
  }
  if ((debug_mode > 1)) {
    printf("load Done\n");
    printf("c_size: %lld, d_size: %lld, l_size: %lld\n", c_size, d_size, l_size);
  }
  fclose(fin);
  // predicted_label = (long long *) calloc(ins_num, sizeof(long long));
  // printf("%lld, %lld， %lld\n", ins_num, l_size, ins_num * l_size);
  // getchar();
  // scores = (real *) calloc(ins_num * l_size, sizeof(real));
}

void LoadValidationData(){
  FILE *fin = fopen(val_file, "r");
  if (fin == NULL) {
    fprintf(stderr, "no such file: %s\n", val_file);
    exit(1);
  }
  if (debug_mode > 1) printf("curInsCount: %lld\n", val_ins_num);
  long long curInsCount = val_ins_num, a, b;
  // if (feof(fin)) printf("EOF!!!\n");

  val_ins = (struct training_ins *) calloc(val_ins_num, sizeof(struct training_ins));
  while(curInsCount--){
    // printf("curInsCount: %lld\n", curInsCount);
    val_ins[curInsCount].id = 1;
    // printf("curInsCount: %lld\n", val_ins[curInsCount].id);
    ReadWord(&val_ins[curInsCount].id, fin);
    // putchar('a');
    ReadWord(&val_ins[curInsCount].c_num, fin);
    ReadWord(&val_ins[curInsCount].sup_num, fin);
    // printf("c_num: %lld, id: %lld, sup_num: %lld\n", val_ins[curInsCount].c_num, val_ins[curInsCount].id, val_ins[curInsCount].sup_num);
    val_ins[curInsCount].cList = (long long *) calloc(val_ins[curInsCount].c_num, sizeof(long long));
    val_ins[curInsCount].supList = (struct supervision *) calloc(val_ins[curInsCount].sup_num, sizeof(struct supervision));
    // printf("%lld, %lld, %lld\n", val_ins[curInsCount].id, val_ins[curInsCount].c_num, val_ins[curInsCount].sup_num);

    for (a = val_ins[curInsCount].c_num; a; --a) {
      ReadWord(&b, fin);
      val_ins[curInsCount].cList[a-1] = b;
      // printf("(%lld)", b);
    }
    // printf("\n");
    for (a = val_ins[curInsCount].sup_num; a; --a) {
      ReadWord(&b, fin);
      val_ins[curInsCount].supList[a-1].label = b;
      ReadWord(&b, fin);
      val_ins[curInsCount].supList[a-1].function_id = b;
      // printf("(%lld, %lld)", val_ins[curInsCount].supList[a-1].label, val_ins[curInsCount].supList[a-1].function_id);
    }
    // printf("\n");
  }
  if ((debug_mode > 1)) {
    printf("load Done\n");
    printf("c_size: %lld, d_size: %lld, l_size: %lld\n", c_size, d_size, l_size);
  }
  fclose(fin);
  // predicted_label = (long long *) calloc(ins_num, sizeof(long long));
  // printf("%lld, %lld， %lld\n", ins_num, l_size, ins_num * l_size);
  // getchar();
  // scores = (real *) calloc(ins_num * l_size, sizeof(real));
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

void LoadModel() {
  FILE *fi = fopen(input_model, "rb");
  long long a, b;
  if (fi == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", input_model);
    exit(1);
  }
  fscanf(fi, "%lld %lld %lld %lld %lld %lld %d %d %d\n", &c_size, &c_length, &l_size, &l_length, &d_size, &NONE_idx, &no_lb, &no_db, &ignore_none);
  if (debug_mode > 1) printf("Initialization\n");
  InitNet();
  
  real sum_check = 0, rsum;
  if (binary) {
    for (b = 0; b < c_size; ++b) {
      for (a = 0; a < c_length; ++a) {
        BREAD(c[b * c_length + a], fi)
        DDMode({printf("%f, ", c[b * c_length + a]);})
        sum_check += c[b* c_length + a];
      }
      DDMode({printf("\n");})
    }
    BREAD(rsum, fi)
    if (sum_check != rsum) {printf("c not fit!\n"); exit(1);}
    sum_check = 0;
    if (0==no_lb) {
      DDMode({printf("wrong!!!\n");})
      for (b = 0; b < l_size; ++b) {
        BREAD(lb[b], fi)
        sum_check += lb[b];
      }
      BREAD(rsum, fi)
      if (sum_check != rsum) {printf("lb not fit!\n"); exit(1);}
      sum_check = 0;
    }
    for (b = 0; b < l_size; ++b) {
      for (a = 0; a < l_length; ++a) {
        BREAD(l[b * l_length + a], fi)
        DDMode({printf("%f, ", l[b * l_length + a]);})
        sum_check += l[b * l_length + a];
      }
      DDMode({printf("\n");})
    }
    BREAD(rsum, fi)
    if (sum_check != rsum) {printf("l not fit!\n"); exit(1);}
    sum_check = 0;
    for (b = 0; b < l_length; ++b) {
      for (a = 0; a < c_length; ++a) {
        BREAD(o[b * c_length + a], fi)
        DDMode({printf("%f, ", o[b * c_length + a]);})
        sum_check += o[b * c_length + a];
        // printf("%lld, %lld, %f, %f\n", b, a, sum_check, o[b * c_length + a]);
      }
      DDMode({printf("\n");})
    }
    BREAD(rsum, fi)
    if (sum_check != rsum) {printf("o not fit!\n"); exit(1);}
    if (debug_mode > 1) printf("sum check pass!\n");
    BREAD(lambda1, fi)
    BREAD(lambda2, fi)
    BREAD(lambda3, fi)
    BREAD(lambda4, fi)
    BREAD(lambda5, fi)
    BREAD(lambda6, fi)
    BREAD(ph1, fi)
    BREAD(ph2, fi)
    for (b = 0; b < c_size; ++b) {
      for (a = 0; a < c_length; ++a) BREAD(cneg[b * c_length + a], fi)
    }
    if (0 == no_db) for (b = 0; b < d_size; ++b) BREAD(db[b], fi)
    for (b = 0; b < d_size; ++b) {
      for (a = 0; a < l_length; ++a) BREAD(d[b * l_length + a], fi)
    }
  }
  fclose(fi);
}


void TruthDiscovery(){
  FILE *fo = fopen(output_file, "w");
  fprintf(fo, "%lld\n", ins_num);
  long long a, b, i, j, l1, label;
  real f, g, h;
  real *cs = (real *) calloc(c_length, sizeof(real));
  real *z = (real *) calloc(l_length, sizeof(real));

  real *score_p = (real *) calloc(l_length, sizeof(real));
  real *score_n = (real *) calloc(l_length, sizeof(real));
  real *label_table = (real *) calloc(l_size, sizeof(real));
  // real *sigmoidD = (real *) calloc(l_length, sizeof(real));
  
  for (i = 0; i < val_ins_num; ++i) {
    struct training_ins * cur_ins = val_ins + i;
    for (j = 0; j < c_length; ++j) cs[j] = 0;
    for (a = 0; a < cur_ins->c_num; ++a) {
      l1 = c_length * cur_ins->cList[a];
      for (j = 0; j < c_length; ++j) cs[j] += c[l1 + j];
    }
    for (j = 0; j < c_length; ++j) cs[j] /= cur_ins->c_num;
    for (a =0; a < l_length; ++a){
      g = 0;
      l1 = a * c_length;
      for (j = 0; j < c_length; ++j) g += cs[j] * o[l1 + j];
#ifdef ACTIVE
      if (g < -MAX_EXP) z[a] = -1;
      else if (g > MAX_EXP) z[a] = 1;
      else z[a] = tanhTable[(int)((g + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
#else
      z[a] = g;
#endif
    }

    fprintf(fo, "[%lld, %lld, [", cur_ins->id, cur_ins->sup_num);

    for (j = 0; j < l_size; ++j) {
      score_n[j] = 0;
      score_p[j] = 0;
      label_table[j] = 0;
    }

    for (b = 0; b < cur_ins->sup_num; ++b){
      j = cur_ins->supList[b].function_id;
      l1 = j * l_length;
      f = 0;
      for (a = 0; a < l_length; ++a) f += z[a] * d[l1 + a];
      if (f > MAX_EXP) g = 1;
      else if (f < -MAX_EXP) g = 0;
      else g = sigTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
      a = cur_ins->supList[b].label;
      if (b > 0) fprintf(fo, ",");
      fprintf(fo, "[%lld, %lld, %f]", j, a, f);
      score_p[a] += log(g * ph1 + (1 - g) * ph2);
      score_n[a] += log(g * (1 - ph1) + (1 - g) * (1 - ph2));
      label_table[a] = 1;
    }

    f = 0.0; for(j = 0; j < l_size; ++j) f += score_n[j];
    g = -INFINITY;
    label = -1;
    for (j = 0; j < l_size; ++j) if ((label_table[j] > 0 ) && (0 == ignore_none || j != NONE_idx)) {
      h = f - score_n[j] + score_p[j];
      if (h > g){
        label = j;
        g = h;
      }
    }

    fprintf(fo, "], %lld]\n", label);
  }
  fclose(fo);
}

int main(int argc, char **argv) {
  srand(19940410);
  int i;
  if (argc == 1) {
    printf("ReHession alpha 1.0\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("-cleng\n-lleng\n-train\n-debug\n-binary\n-alpha\n-reSample\n-sample\n-negative\n-threads\n-min-count\n-instances\n-infer_together\n-alpha_update_every\n-iter\n-none_idx\n-no_lb\n-no_db\n-lambda1\n-lambda2\n-grad_clip\n-ingore_none\n-error_log\n-normL\n-dropout(D Mode)\nlambda1: skip-gram\nlambda2: truth finding\nlambda3: l\nlambda4: d\nlambda5: o\n lambda6: c\n");
    // printf("\t-none_idx <file>\n");
    // printf("\t\tthe index of None Type\n");
    printf("\nExamples:\n");
    printf("./rmodify -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/test.data -threads 20 -NONE_idx 6 -cleng 30 -lleng 50 -resample 30 -ignore_none 0 -iter 100 -normL 0 -debug 2 -dropout 0.5\n\n");//-none_idx 5 
    return 0;
  }
  test_file[0] = 0;
  val_file[0] = 0;
  input_model[0] = 0;
  output_file[0] = 0;
  // save_vocab_file[0] = 0;
  // read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-cleng", argc, argv)) > 0) c_length = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-lleng", argc, argv)) > 0) l_length = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-special_none", argc, argv)) > 0) special_none = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-useEntropy", argc, argv)) > 0) useEntropy = atoi(argv[i + 1]);
  // if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(input_model, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-test", argc, argv)) > 0) strcpy(test_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-val", argc, argv)) > 0) strcpy(val_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-input_model", argc, argv)) > 0) strcpy(input_model, argv[i + 1]);
  if ((i = ArgPos((char *)"-output_file", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-reSample", argc, argv)) > 0) reSample = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-instances", argc, argv)) > 0) ins_num = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-test_instances", argc, argv)) > 0) test_ins_num = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-val_instances", argc, argv)) > 0) val_ins_num = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-infer_together", argc, argv)) > 0) infer_together = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha_update_every", argc, argv)) > 0) print_every = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-print_val", argc, argv)) > 0) printVal = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iters = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-none_idx", argc, argv)) > 0) NONE_idx = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-no_lb", argc, argv)) > 0) no_lb = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-no_db", argc, argv)) > 0) no_db = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-normL", argc, argv)) > 0) normL = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-ignore_none", argc, argv)) > 0) ignore_none = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda1", argc, argv)) > 0) lambda1 = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda2", argc, argv)) > 0) lambda2 = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda3", argc, argv)) > 0) lambda3 = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda4", argc, argv)) > 0) lambda4 = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda5", argc, argv)) > 0) lambda5 = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda6", argc, argv)) > 0) lambda6 = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-grad_clip", argc, argv)) > 0) grad_clip = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-error_log", argc, argv)) > 0) error_log = atoi(argv[i + 1]);
#ifdef DROPOUT
  if ((i = ArgPos((char *)"-dropout", argc, argv)) > 0) dropout = atof(argv[i + 1]) * DROPOUTRATIO;
#endif
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  sigTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  tanhTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  if (sigTable == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  if (debug_mode > 1) printf("Starting\n");
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    sigTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    tanhTable[i] = (expTable[i] * expTable[i] - 1)/(expTable[i] * expTable[i] + 1);
  }

  // if (debug_mode > 1) printf("Loading training file %s\n", input_model);
  // LoadTrainingData();
  if (debug_mode > 1) printf("Truth Discovery \n ");
  LoadModel();
  // if (debug_mode > 1) printf("\nLoading test file %s\n", test_file);
  // LoadTestingData();
  if (debug_mode > 1) printf("\nLoading validation file %s\n", val_file);
  LoadValidationData();
  if (debug_mode > 1) printf("Truth Discovery \n ");
  TruthDiscovery();
  if (debug_mode > 1) printf("releasing memory\n");
  DestroyNet();
  free(expTable);
  free(sigTable);
  free(tanhTable);
  return 0;
}