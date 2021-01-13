//
// Created by xinyan on 9/1/2020.
//

#pragma once
#include <iostream>
#include <vector>
using std::string;
using std::min;
using std::max;
using std::vector;

int edit_distance(const string & a, const string& b) {
  int na = (int) a.size();
  int nb = (int) b.size();

  vector<int> f(nb+1, 0);
  for (int j = 1; j <= nb; ++j) {
    f[j] = j;
  }

  for (int i = 1; i <= na; ++i) {
    int prev = i;
    for (int j = 1; j <= nb; ++j) {
      int cur;
      if (a[i-1] == b[j-1]) {
        cur = f[j-1];
      }
      else {
        cur = min(min(f[j-1], prev), f[j]) + 1;
      }

      f[j-1] = prev;
      prev = cur;
    }
    f[nb] = prev;
  }
  return f[nb];
}


#define PRUNE_K  (20000)

typedef int64_t int64;
typedef int32_t int32;
// #define max(x,y)  (((x)>(y))?(x):(y))
// #define min(x,y)  (((x)<(y))?(x):(y))

int slide(const char *x, const char *y)
// computes slide of x and y - returns the index of the first disagreement
// x and y should be zero terminted plus there should be 7 characters afterwards that are different for x and y
// the procedure assumes little-endian memory layout for integers (intel x86 type).
// For big-endian one would need to modify the computation using __builtin_ctz to something using __builtin_clz
{
  int i=0;

//	printf("(%s,%s,",x,y);

  while(*((int64*)x)==*((int64*)y)){ x+=8; y+=8; i+=8; }		// find the first 8-bytes that differ

//	printf("%i --- %lx --- (%i) ",i,(*((int64*)x)-*((int64*)y)), __builtin_ctz(*((int64*)x)-*((int64*)y)));

  i+= (__builtin_ctzll(*((int64*)x)-*((int64*)y)) >> 3);	// calculates the first byte of the 8 that differs

//	printf("%i) ",i);

  return i;
}

int slide32(const char *x, const char *y)
// computes slide of x and y - returns the index of the first disagreement
// x and y should be zero terminted plus there should be 7 characters afterwards that are different for x and y
// the procedure assumes little-endian memory layout for integers (intel x86 type).
// For big-endian one would need to modify the computation using __builtin_ctz to something using __builtin_clz
{
  int i=0;

//	printf("(%s,%s,",x,y);

  while(*((int32*)x)==*((int32*)y)){ x+=4; y+=4; i+=4; }		// find the first 8-bytes that differ

//	printf("%i --- %lx --- (%i) ",i,(*((int32*)x)-*((int32*)y)), __builtin_ctz(*((int32*)x)-*((int32*)y)));

  i+= (__builtin_ctz(*((int32*)x)-*((int32*)y)) >> 3);	// calculates the first byte of the 8 that differs

//	printf("%i) ",i);

  return i;
}


int edit_distance(const char *x, const int x_len, const  char *y, const int y_len, int k)
// computes the edit distance of x and y
// x and y should be zero terminated plus there should be 7 characters afterwards that are different for x and y
// (we don't really need zero termination but we need 8 characters after x and y that differ)
{
  if(k >= PRUNE_K)return -2;			// error - too large k

  if(x_len > y_len)return edit_distance(y,y_len,x,x_len,k);

  int fc_buf[2*PRUNE_K+1],fp_buf[2*PRUNE_K+1];        // must be at least 2k+3
  int *fc,*fp;				// current F(d,h) and previous F(d,h-1)
  int h,dl,du,d;

  fc_buf[PRUNE_K]=fp_buf[PRUNE_K]=-1;

  for(h=0;h <= k; h++){

    if( (h&1)==0 ){ fc=fc_buf+PRUNE_K; fp=fp_buf+PRUNE_K; }else{ fc=fp_buf+PRUNE_K; fp=fc_buf+PRUNE_K; }

    dl = - min( 1+((k-(y_len-x_len))/2), h);	// compute the range of relevant diagonals
    du =   min( 1+k/2+(y_len-x_len), h);

    fp[dl-1]=fp[dl]=fp[du]=fp[du+1]=-1;

    for(d= dl;d <= du; d++){
      int r=max(max(fp[d-1],fp[d]+1),fp[d+1]+1);

      if((r >= x_len) || (r+d >= y_len))fc[d]=r;
      else fc[d]=r+slide(x+r,y+r+d);

      if((d== y_len-x_len)&&(fc[d]>=x_len))return h;
    }
  }
  return -1;
}
