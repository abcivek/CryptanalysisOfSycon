/*
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>
*/
#ifndef SYCON_H_
#define SYCON_H_

#include <vector>
#include <array>
#include <random>
#include <iostream>
#include <iomanip>

#include "layer.h"
#include "mask.h"
#include "statemask.h"
#include "step_linear.h"
#include "step_nonlinear.h"
#include "lrucache.h"


struct SyconState : public StateMask<5,64> {
  SyconState();
  friend std::ostream& operator<<(std::ostream& stream, const SyconState& statemask);
  void print(std::ostream& stream);
  virtual SyconState* clone();
};


#define ROTR(x,n) (((x)>>(n))|((x)<<(64-(n))))
#define ROTL(x,n) (((x)<<(n))|((x)>>(64-(n))))

template <unsigned round>
std::array<BitVector, 1> SyconSigma(std::array<BitVector, 1> in) {
  switch (round) {
    case 0: 
      return {ROTL(in[0] ^ ROTL(in[0], 59) ^ ROTL(in[0], 54),40)};
    case 1:
      return {ROTL(in[0] ^ ROTL(in[0], 55) ^ ROTL(in[0], 46),32)};
    case 2: 
      return { ROTL(in[0] ^ ROTL(in[0], 33) ^ ROTL(in[0], 2),16)};
    case 3: 
      return { ROTL(in[0] ^ ROTL(in[0], 21) ^ ROTL(in[0], 42),56)};
    case 4:
      return {in[0] ^ ROTL(in[0],  13) ^ ROTL(in[0], 26)};
    default: 
      return {0};
  }
}

struct SyconLinearLayer : public LinearLayer {
  SyconLinearLayer& operator=(const SyconLinearLayer& rhs);
  SyconLinearLayer();
  virtual SyconLinearLayer* clone();
  void Init();
  SyconLinearLayer(StateMaskBase *in, StateMaskBase *out);
//  bool Update();
  virtual bool updateStep(unsigned int step_pos);
  unsigned int GetNumSteps();
  virtual void copyValues(LinearLayer* other);

  static const unsigned int word_size_ = { 64 };
  static const unsigned int words_per_step_ = { 1 };
  static const unsigned int linear_steps_ = { 5 };
  std::array<LinearStep<word_size_, words_per_step_>, linear_steps_> sigmas;
};


struct SyconSboxLayer : public SboxLayer<5, 64> {
  SyconSboxLayer& operator=(const SyconSboxLayer& rhs);
  SyconSboxLayer();
  SyconSboxLayer(StateMaskBase *in, StateMaskBase *out);
  virtual SyconSboxLayer* clone();
//  virtual bool Update();
  virtual bool updateStep(unsigned int step_pos);
  Mask GetVerticalMask(unsigned int b, const StateMaskBase& s) const;
  void SetVerticalMask(unsigned int b, StateMaskBase& s, const Mask& mask);

 static const unsigned int cache_size_ = { 0x1000 };
 static std::unique_ptr<LRU_Cache<unsigned long long,NonlinearStepUpdateInfo>> cache_;
 static std::shared_ptr<LinearDistributionTable<5>> ldt_;
};



#endif // SYCON_H_
