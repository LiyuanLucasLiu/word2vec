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
#define BWRITE(x,f) fwrite(&x , sizeof(real), 1, f);
#define SWRITE(x,f) fprintf(f, "%lf ", x);

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

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
// struct vocab_word *vocab;
long long  *cCount;
int binary = 1, debug_mode = 2, reSample = 10, min_count = 5, num_threads = 1, min_reduce = 1;
long long c_size = 0, c_length = 100, l_size = 1, l_length = 400, d_size, NONE_idx, tot_c_count = 0;
real lambda1 = 0.3, lambda2 = 0.3;
long long ins_num = 181965, ins_count_actual = 0;
long long iters = 10;
long print_every = 1000;
real alpha = 0.025, starting_alpha, sample = 1e-4;

struct training_ins * data;
real *c, *l, *d, *cneg, *db, *lb;
real *o;
real *ph1, *ph2;
real *sigTable;
clock_t start;

int negative = 5;
const int table_size = 1e8;
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
      default:
        *word = *word * 10 + ch - '0';
    }// Truncate too long words
  }
  return 0;
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
  a = posix_memalign((void **)&lb, 128, (long long)l_size * sizeof(real));
  CHECKNULL(lb)
  a = posix_memalign((void **)&db, 128, (long long)d_size * sizeof(real));
  CHECKNULL(db)
  cCount = (long long *) calloc(c_size,  sizeof(long long));
  CHECKNULL(cCount)
  memset(cCount, 0, c_size);
  ph1 = (real*) calloc(d_size, sizeof(real));
  CHECKNULL(ph1)
  ph2 = (real*) calloc(d_size, sizeof(real));
  CHECKNULL(ph2)

  for (b = 0; b < c_size; ++b) for (a = 0; a < c_length; ++a) {
    c[b * c_length + a] = (rand() / (real)RAND_MAX - 0.5) / c_length;
    cneg[b * c_length + a] = 0;
  }
  // printf("%lld\n", b* c_length + a);
  // getchar();
  for (b = 0; b < l_size; ++b) lb[b] = 0;
  for (b = 0; b < d_size; ++b) db[b] = 0;
  for (b = 0; b < d_size; ++b) ph1[b] = 0.5;
  for (b = 0; b < d_size; ++b) ph2[b] = 0.5;
  for (b = 0; b < l_size; ++b) for (a = 0; a < l_length; ++a)
    l[b * l_length + a] = 0;//(rand() / (real)RAND_MAX - 0.5) / l_length;
  for (b = 0; b < d_size; ++b) for (a = 0; a < l_length; ++a)
    d[b * l_length + a] = 0;//(rand() / (real)RAND_MAX - 0.5) / d_length; 
  for (b = 0; b < l_length; ++b) for (a = 0; a < c_length; ++a)
    o[b * c_length + a] = 0; 
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

void LoadTrainingData(){
  FILE *fin = fopen(train_file, "r");
  if (fin == NULL) {
    fprintf(stderr, "no such file: %s\n", train_file);
    exit(1);
  }
  printf("curInsCount: %lld\n", ins_num);
  long long curInsCount = ins_num, a, b;
  
  data = (struct training_ins *) calloc(ins_num, sizeof(struct training_ins));
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

void *TrainModelThread(void *id) {
  unsigned long long next_random = (long long)id;
  long long cur_iter = 0, end_id = ((long long)id+1) * ins_num / num_threads;
  long long cur_id, last_id;
  clock_t now;
  long long a, b, i, j, l1, l2;
  real f, g, h;
  long long target, label;
  struct training_ins *cur_ins;
  real *c_error = (real *) calloc(c_size, sizeof(real));
  real *z = (real *) calloc(l_size, sizeof(real));
  real *z_error = (real *) calloc(l_size, sizeof(real));
  real *score_p = (real *) calloc(l_size, sizeof(real));
  real *score_n = (real *) calloc(l_size, sizeof(real));
  real *score_kl = (real *) calloc(l_size, sizeof(real));
  real *sigmoidD = (real *) calloc(l_size, sizeof(real));
  real sum_softmax;
  struct training_ins tmpIns;
  while (cur_iter < iters) {
    cur_id = (long long)id * ins_num / num_threads;
    last_id = cur_id;
    while(cur_id < end_id){
      // update threads
      if (cur_id - last_id > 1000) {
        ins_count_actual += cur_id - last_id;
        last_id = cur_id;
        if ((debug_mode > 1)) {
          now = clock();
          printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
            ins_count_actual / (real)(ins_num * iters + 1) * 100,
            ins_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
          // fflush(stdout);
        }
        alpha = starting_alpha * (1 - ins_count_actual) / (real) (ins_num * iters + 1);
        if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
      }

      cur_ins = data + cur_id;

      // feature embedding learning
      for (i = 0; i < reSample; ++i){
        for (b = -1; b < 0; ){
          if (b != -2 && sample > 0) {
            //down sampling
            NRAND
            b = next_random % cur_ins->c_num;
            // printf("b: %lld\n", b);
            b = cur_ins->cList[b];
            // printf("bNum: %lld\n", b);
            real ran = (sqrt(cCount[b] / (sample * tot_c_count)) + 1) * (sample * tot_c_count) / cCount[b];
            // printf("ran: %f\n", ran);
            NRAND
            // printf("nr: %lld", (next_random & 0xFFFF));
            if (ran < (next_random & 0xFFFF) / (real)65536) b = -2;
          } else {
            NRAND
            b = next_random % cur_ins->c_num;
            b = cur_ins->cList[b];
          }
        }
        l1 = b * c_length;
        for (a = 0; a < c_length; ++a) c_error[a] = 0.0;
        for (j = 0; j < negative + 1; ++j){
          NRAND
          if (0 == j){
            //pos
            target = next_random % cur_ins->c_num;
            label = 1;
          } else {
            //neg
            target = table[(next_random >> 16) % table_size];
            label = 0;
          }
          if (target == b) continue;
          l2 = target * c_length;
          f = 0.0;
          for (a = 0; a < c_length; ++a) f += c[a + l1] * cneg[a + l2];
          // printf("f: %f, l1: %lld, l2: %lld\n", f, l1, l2);
          if (f > MAX_EXP) g = (label - 1) * alpha * lambda1;
          else if (f < -MAX_EXP) g = (label - 0) * alpha * lambda1;
          else {
            // printf("%f\n", f);
            g = (label - sigTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha * lambda1;
            // printf("%f\n", g);
          }
          // putchar('c');
          for (a = 0; a < c_length; ++a) c_error[a] += g * cneg[a + l2];
          // putchar('c');
          for (a = 0; a < c_length; ++a) cneg[a + l2] += g * c[a + l1];
          // putchar('\n');
        }
        for (a = 0; a < c_length; ++a) c[a + l1] += c_error[a];
      }

      //cal relation mention embedding
      for (a = 0; a < c_length; ++a) c_error[a] = 0.0;
      for (i = 0; i < cur_ins->c_num; ++i) {
        l1 = cur_ins->cList[i] * c_length;
        for (j = 0; j < c_length; ++j) c_error[j] += c[l1 + j];
      }
      for (a = 0; a < c_length; ++a) c_error[a] /= i;
      for (a = 0; a < l_length; ++a) {
        z[a] = 0.0; z_error[a] = 0.0;
        l1 = a * c_length;
        for (i = 0; i < c_length; ++i) z[a] += c_error[i] * o[l1 + i];
      }

      // infer true labels
      // score here is prob, which is the larger the better
      for (i = 0; i < l_size; ++i) {
        score_n[i] = 0;
        score_p[i] = 0;
      }
      for (i = 0 ; i < cur_ins->sup_num ; ++i){
        j = cur_ins->supList[i].function_id;
        l1 = j * l_length; 
        f = db[j]; 
        for (a = 0; a < l_length; ++a) f+= z[a] * d[l1 + a];
        if (f > MAX_EXP) g = lambda2;
        else if (f < -MAX_EXP) g = -lambda2;
        else g = sigTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))] * lambda2;
        a = cur_ins->supList[i].label;
        sigmoidD[a] = g;
        score_p[a] += log(g * ph1[j] + (1 - g) * ph2[j]);
        score_n[a] += log(g * (1 - ph1[j]) + (1 - g) * (1 - ph2[j]));
      }
      sum_softmax = 0.0;
      g = 0.0;
      for (i = 0 ; i < l_size; ++i) if (i!= NONE_idx) {
        f = lb[i];
        l1 = i * l_length;
        for (a = 0; a < l_length; ++a) f += z[a] * l[l1 + a];
        score_p[i] += f; //max
        g += f;
        score_kl[i] = exp(f);
        sum_softmax += score_kl[i];
      }
      g /= (l_size - 1);
      score_p[NONE_idx] += g + log(l_size-1); //max
      g = log(sum_softmax);
      for (i = 0; i < l_size; ++i) {
        score_p[i] -= g; // max
      }
      f = 0.0; for (i = 0; i < l_size; ++i) f += score_n[i];
      g = f - score_n[NONE_idx] + score_p[NONE_idx];
      label = NONE_idx;
      for (i = 0; i < l_size; ++i) {
        h = f - score_n[i] + score_p[i];
        if (h > g){
          label = i;
          g = h;
        }
      }
      // update params 
      // update l, lb
      printf("0:%lf, %lf\n", z_error[0], o[0]);
      if (label == NONE_idx) {
        putchar('c');
        putchar('\n');
        for (i = 0; i < l_size; ++i) if (i != NONE_idx) {
          l1 = i * l_length;
          f = alpha * score_kl[i] / sum_softmax;
          lb[i] += alpha / (l_size - 1) - f;
          for (a = 0; a < l_length; ++a) {
            z_error[a] += alpha / (l_size-1) - l[l1 + a] * f;
            l[l1 + a] += alpha / (l_size-1) - z[a] * f;
          }
        } 
      } else {
        l1 = label * l_length;
        f = alpha * score_kl[label] / sum_softmax;
        printf("%f, %f, %f, %f\n",l[l1], z[0], z_error[0], f);
        lb[label] += alpha  - f;
        for (a = 0; a < l_length; ++a){
          z_error[a] += l[l1 + a] * (alpha - f);
          l[l1 + a] += z[a] * (alpha - f) ;
        }
        printf("%f\n", z_error[0]);
        for (i = 0; i < l_size; ++i) if (i != NONE_idx && i != label) {
          l1 = i * l_length;
          f = alpha * score_kl[i] / sum_softmax;
          lb[i] -= f;
          for (a = 0; a < l_length; ++a) {
            z_error[a] -= l[l1 + a] * f;
            l[l1 + a] -= z[a] * f;
          }
        }
      }
      // update d, db
      // update ph1, ph2
      printf("1:%lf, %lf\n", z_error[0], o[0]);
      for (i = 0 ; i < cur_ins->sup_num ; ++i){
        j = cur_ins->supList[i].function_id;
        a = cur_ins->supList[i].label;
        f = log(sigmoidD[a] * ph1[j] + (1 - sigmoidD[a]) * ph2[j]);
        if (a == label) {
          //d, db
          g = alpha * lambda2 * (ph1[j] - ph2[j]) * sigmoidD[a] * (1- sigmoidD[a]) / f;
          l1 = j * l_length;
          db[j] += g;
          for (b = 0; b < l_length; ++b){
            z_error[b] += d[l1 + b] * g;
            d[l1 + b] += z[b] * g;
          }
          //ph1, ph2
          ph1[j] += alpha * lambda2 * sigmoidD[a] / f;
          ph1[j] = ph1[j] > 0 ? ph1[j] : 0;
          ph1[j] = ph1[j] < 1 ? ph1[j] : 1;
          ph2[j] += alpha * lambda2 * (1 - sigmoidD[a]) / f;
          ph2[j] = ph2[j] > 0 ? ph2[j] : 0;
          ph2[j] = ph2[j] < 1 ? ph2[j] : 1;
        } else {
          //d, db
          g = alpha * lambda2 * (ph2[j] - ph1[j]) * sigmoidD[a] * (1 - sigmoidD[a]) / f;
          l1 = j * l_length;
          db[j] += g;
          for (b = 0; b< l_length; ++b) {
            z_error[b] += d[l1 + b] * g;
            d[l1 + b] += z[b] * g;
          }
          //ph1, ph2
          ph1[j] -= alpha * lambda2 * sigmoidD[a] / f;
          ph2[j] -= alpha * lambda2 * (1 - sigmoidD[a]) / f;
        }
      }
      // update o
      printf("2:%lf, %lf\n", z_error[0], o[0]);
      for (a = 0; a < l_length; ++a) {
        l1 = a * c_length;
        for (b = 0; b < c_length; ++b) o[l1 + b] += z_error[a] * c_error[b];
      }
      // update c
      for (a = 0; a < c_length; ++a) c_error[a] = 0;
      for (a = 0; a < l_length; ++a) {
        l1 = a * c_length;
        // if (a % 40 ==0 ) printf("%f, %f, %f, %f, %lld\n", c[9232900], c_error[0], z_error[a], o[l1], a);
        for (b = 0; b < c_length; ++b) c_error[b] += z_error[a] * o[l1 + b];
      }
      for (a = 0; a < c_length; ++a) c_error[a] /= cur_ins->c_num;
      for (i = 0; i < cur_ins->c_num; ++i) {
        l1 = cur_ins->cList[i] * c_length;
        for (j = 0; j < c_length; ++j) c[l1 + j] += c_error[j];
      }
      // update index
      ++cur_id;
    }
    //shuffle
    for (cur_id = (long long)id * ins_num / num_threads; cur_id < end_id - 1; ++cur_id){
      a = end_id - cur_id;
      copyIns(&tmpIns, data + cur_id);
      NRAND
      b = next_random % a;
      copyIns(data + cur_id, data + cur_id + b);
      copyIns(data + cur_id + b, &tmpIns);
    }
  }
  pthread_exit(NULL);
}

void TrainModel() {
  long long a, b;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  if (pt == NULL) {
    fprintf(stderr, "cannot allocate memory for threads\n");
    exit(1);
  }
  printf("Starting training using threads %d\n", num_threads);
  starting_alpha = alpha;
  tot_c_count = 0;
  memset(cCount, 0, c_size);
  if (negative > 0) {
    for (a = 0; a < ins_num; ++a) {
      for (b = data[a].c_num; b; --b) {
        ++cCount[data[a].cList[b-1]];
        ++tot_c_count;
      }
    }
    InitUnigramTable();
  }
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  free(table);
  free(pt);
}

void SaveModel() {  
  FILE *fo = fopen(output_file, "wb");
  long long a, b;
  if (fo == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", output_file);
    exit(1);
  }
  fprintf(fo, "%lld %lld %lld %lld %lld %lld\n", c_size, c_length, l_size, l_length, d_size, NONE_idx);
  if (binary) {
    BWRITE(lambda1, fo)
    BWRITE(lambda2, fo)
    fprintf(fo, "\n");
    for (b = 0; b < c_size; ++b) {
      for (a = 0; a < c_length; ++a) BWRITE(c[b * c_length + a], fo)
      fprintf(fo, "\n");
    }
    for (b = 0; b < c_size; ++b) {
      for (a = 0; a < c_length; ++a) BWRITE(cneg[b * c_length + a], fo)
      fprintf(fo, "\n");
    }
    for (b = 0; b < l_size; ++b) BWRITE(lb[b], fo)
    fprintf(fo, "\n"); 
    for (b = 0; b < d_size; ++b) BWRITE(db[b], fo)
    fprintf(fo, "\n");
    for (b = 0; b < d_size; ++b) BWRITE(ph1[b], fo)
    fprintf(fo, "\n");
    for (b = 0; b < d_size; ++b) BWRITE(ph2[b], fo)
    fprintf(fo, "\n");
    for (b = 0; b < l_size; ++b) {
      for (a = 0; a < l_length; ++a) BWRITE(l[b * l_length + a], fo)
      fprintf(fo, "\n");
    }
    for (b = 0; b < d_size; ++b) {
      for (a = 0; a < l_length; ++a) BWRITE(d[b * l_length + a], fo)
      fprintf(fo, "\n");
    }
    for (b = 0; b < l_length; ++b) {
      for (a = 0; a < c_length; ++a) BWRITE(o[b * c_length + a], fo)
      fprintf(fo, "\n");
    }
  } else {
    SWRITE(lambda1, fo)
    SWRITE(lambda2, fo)
    fprintf(fo, "\n");
    for (b = 0; b < c_size; ++b) {
      for (a = 0; a < c_length; ++a) SWRITE(c[b * c_length + a], fo)
      fprintf(fo, "\n");
    }
    for (b = 0; b < c_size; ++b) {
      for (a = 0; a < c_length; ++a) SWRITE(cneg[b * c_length + a], fo)
      fprintf(fo, "\n");
    }
    for (b = 0; b < l_size; ++b) SWRITE(lb[b], fo)
    fprintf(fo, "\n"); 
    for (b = 0; b < d_size; ++b) SWRITE(db[b], fo)
    fprintf(fo, "\n");
    for (b = 0; b < d_size; ++b) SWRITE(ph1[b], fo)
    fprintf(fo, "\n");
    for (b = 0; b < d_size; ++b) SWRITE(ph2[b], fo)
    fprintf(fo, "\n");
    for (b = 0; b < l_size; ++b) {
      for (a = 0; a < l_length; ++a) SWRITE(l[b * l_length + a], fo)
      fprintf(fo, "\n");
    }
    for (b = 0; b < d_size; ++b) {
      for (a = 0; a < l_length; ++a) SWRITE(d[b * l_length + a], fo)
      fprintf(fo, "\n");
    }
    for (b = 0; b < l_length; ++b) {
      for (a = 0; a < c_length; ++a) SWRITE(o[b * c_length + a], fo)
      fprintf(fo, "\n");
    }
  }
  fclose(fo);
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
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
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
    printf("\t-instances <int>\n");
    printf("\t\tthe number of instances in training set\n");
    printf("\t-iter <file>\n");
    printf("\t\tnumber of iters; default is 10\n");
    printf("\t-alpha_update_every <file>\n");
    printf("\t\tprint every # of instances; default is 1000\n");
    printf("\t-none_idx <file>\n");
    printf("\t\tthe index of None Type\n");
    printf("\nExamples:\n");
    printf("./recol -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -output /shared/data/ll2/CoType/data/intermediate/KBP/default.model -none_idx 5 \n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-cleng", argc, argv)) > 0) c_length = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-lleng", argc, argv)) > 0) l_length = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-reSample", argc, argv)) > 0) reSample = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-instances", argc, argv)) > 0) ins_num = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha_update_every", argc, argv)) > 0) print_every = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-none_idx", argc, argv)) > 0) NONE_idx = atoi(argv[i + 1]);
  else {fprintf(stderr, "none_idx is required" );}
  if ((i = ArgPos((char *)"-lambda1", argc, argv)) > 0) lambda1 = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda2", argc, argv)) > 0) lambda2 = atof(argv[i + 1]);
  // expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  sigTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  if (sigTable == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    sigTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    sigTable[i] = sigTable[i] / (sigTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  printf("Loading training file %s\n", train_file);
  LoadTrainingData();
  printf("Initialization\n");
  InitNet();
  printf("start training \n ");
  TrainModel();
  printf("Saving to %s\n", output_file);
  SaveModel();
  printf("releasing memory");
  DestroyNet();
  // free(expTable);
  free(sigTable);
  return 0;
}
