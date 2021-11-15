/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2016-2021 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "avx_impl.h"
#include "avx-pairhmm.h"

Context<float> g_ctxf;
Context<double> g_ctxd;

float compute_avxs(testcase *tc)
{
  float result = compute_full_prob_avxs<float>(tc);
  return log10f(result) - g_ctxf.LOG10_INITIAL_CONSTANT;
}

double compute_avxd(testcase *tc)
{
  double result = compute_full_prob_avxd<double>(tc);
  return log10(result) - g_ctxd.LOG10_INITIAL_CONSTANT;
}

double compute_avx(testcase *tc)
{
  double result_final = 0;
  float result_float = compute_full_prob_avxs<float>(tc);

  if (result_float < MIN_ACCEPTED) {
    double result_double = compute_full_prob_avxd<double>(tc);
    result_final = log10(result_double) - g_ctxd.LOG10_INITIAL_CONSTANT;
  }
  else {
    result_final = (double)(log10f(result_float) - g_ctxf.LOG10_INITIAL_CONSTANT);
  }
  return result_final;
}
