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
#include "sycon.h"

//#define HEXOUTPUT
//#define LATEX
//#define TRUNCATED

SyconState::SyconState()
    : StateMask() {
}

SyconState* SyconState::clone() {
  SyconState* obj = new SyconState();
  for (size_t j = 0; j < words_.size(); ++j) {
    obj->words_[j] = words_[j];
    obj->changes_for_linear_layer_[j] = changes_for_linear_layer_[j];
    obj->changes_for_sbox_layer_[j] = changes_for_sbox_layer_[j];
  }
  return obj;
}

void SyconState::print(std::ostream& stream) {
  stream << *this;
}


#ifdef HEXOUTPUT
#ifdef LATEX
std::ostream& operator<<(std::ostream& stream, const SyconState& statemask) {

  for (size_t i = 0; i < statemask.words_.size(); ++i) {
    unsigned long long outword = 0;
    for (auto it = statemask.words_[i].bitmasks.rbegin(); it != statemask.words_[i].bitmasks.rend(); ++it) {
      outword <<= 1;
      if (*it % 4 == 1)
        outword |=1;
    }
    stream << " & \\texttt{" << std::hex << std::setfill('0') << std::setw(16) << outword << "} " << std::dec;
  }
  stream << std::endl;
  return stream;
}
#else
#ifdef TRUNCATED
std::ostream& operator<<(std::ostream& stream, const SyconState& statemask) {

  unsigned long long resultingword = 0;
  for (size_t i = 0; i < statemask.words_.size(); ++i) {
    unsigned long long outword = 0;
    for (auto it = statemask.words_[i].bitmasks.rbegin(); it != statemask.words_[i].bitmasks.rend(); ++it) {
      outword <<= 1;
      if (*it % 4 == 1)
        outword |=1;
    }
    resultingword |= outword;
  }
  stream << std::hex << std::setfill('0') << std::setw(16) << resultingword << std::dec <<  std::endl;
  return stream;
}
#else
std::ostream& operator<<(std::ostream& stream, const SyconState& statemask) {

  for (size_t i = 0; i < statemask.words_.size(); ++i) {
    unsigned long long outword = 0;
    for (auto it = statemask.words_[i].bitmasks.rbegin(); it != statemask.words_[i].bitmasks.rend(); ++it) {
      outword <<= 1;
      if (*it % 4 == 1)
        outword |=1;
    }
    stream << std::hex << std::setfill('0') << std::setw(16) << outword << std::dec <<  std::endl;
  }
  return stream;
}
#endif
#endif
#else
std::ostream& operator<<(std::ostream& stream, const SyconState& statemask) {
#ifndef TERMINALCOLORS
  char symbol[4] {'#', '1', '0', '?'};
#else
  std::string symbol[4] { "\033[1;35m#\033[0m", "\033[1;31m1\033[0m", "0",
      "\033[1;33m?\033[0m" };
#endif
  for (Mask word : statemask.words_) {
    for (auto it = word.bitmasks.rbegin(); it != word.bitmasks.rend(); ++it) {
      stream << symbol[*it % 4];
    }
    stream << std::endl;
  }
  return stream;
}
#endif

//-----------------------------------------------------------------------------

SyconLinearLayer& SyconLinearLayer::operator=(const SyconLinearLayer& rhs) {
  sigmas = rhs.sigmas;
  return *this;
}

SyconLinearLayer::SyconLinearLayer() {
  Init();
}

unsigned int SyconLinearLayer::GetNumSteps() {
  return linear_steps_;
}

SyconLinearLayer* SyconLinearLayer::clone() {
  //TODO: write copy constructor
  SyconLinearLayer* obj = new SyconLinearLayer(in, out);
  obj->sigmas = this->sigmas;
  return obj;
}

SyconLinearLayer::SyconLinearLayer(StateMaskBase *in, StateMaskBase *out)
    : LinearLayer(in, out) {
  Init();
}

void SyconLinearLayer::Init() {
  sigmas[0].Initialize(SyconSigma<0>);
  sigmas[1].Initialize(SyconSigma<1>);
  sigmas[2].Initialize(SyconSigma<2>);
  sigmas[3].Initialize(SyconSigma<3>);
  sigmas[4].Initialize(SyconSigma<4>);
}


bool SyconLinearLayer::updateStep(unsigned int step_pos) {

  bool ret_val =  sigmas[step_pos].Update( { &((*in)[step_pos]) },
                                   { &((*out)[step_pos]) });

  return ret_val;
}

void SyconLinearLayer::copyValues(LinearLayer* other){
  SyconLinearLayer* ptr = dynamic_cast<SyconLinearLayer*> (other);
  sigmas = ptr->sigmas;
}

//-----------------------------------------------------------------------------

BitVector SyconSbox(BitVector in) {
  // with x0 as LSB 
   /* 
 static const BitVector sbox[32] = {
  
       8, 19, 30, 7, 6, 25, 16, 13, 22, 15, 3, 24, 17, 12, 4, 27,
       11, 0, 29, 20, 1, 14, 23, 26, 28, 21, 9, 2, 31, 18, 10, 5,

  };*/
  // with x0 as MSB 
  static const BitVector sbox[32] = {
  
       2, 26, 13, 7, 12, 16, 17, 31, 15, 23, 24, 18, 1, 29, 4, 10, 
       25, 0, 30, 21, 19, 14, 6, 9, 28, 5, 3, 8, 22, 11, 27, 20

  };
  return sbox[in % 32] & 0x1f;
}


std::unique_ptr<LRU_Cache<unsigned long long, NonlinearStepUpdateInfo>> SyconSboxLayer::cache_;
std::shared_ptr<LinearDistributionTable<5>> SyconSboxLayer::ldt_;

SyconSboxLayer& SyconSboxLayer::operator=(const SyconSboxLayer& rhs) {
  sboxes = rhs.sboxes;
  return *this;
}

SyconSboxLayer::SyconSboxLayer() {
  if(ldt_ == nullptr)
    ldt_.reset(new LinearDistributionTable<5>(SyconSbox));
  InitSboxes(ldt_);
  if (this->cache_.get() == nullptr)
    this->cache_.reset(
        new LRU_Cache<unsigned long long, NonlinearStepUpdateInfo>(
            cache_size_));
}

SyconSboxLayer::SyconSboxLayer(StateMaskBase *in, StateMaskBase *out)
    : SboxLayer(in, out) {
  if(ldt_ == nullptr)
    ldt_.reset(new LinearDistributionTable<5>(SyconSbox));
  InitSboxes(ldt_);
  if (this->cache_.get() == nullptr)
    this->cache_.reset(
        new LRU_Cache<unsigned long long, NonlinearStepUpdateInfo>(
            cache_size_));
}

SyconSboxLayer* SyconSboxLayer::clone() {
  //TODO: write copy constructor
  SyconSboxLayer* obj = new SyconSboxLayer(in, out);
  obj->sboxes = this->sboxes;
  return obj;
}


bool SyconSboxLayer::updateStep(unsigned int step_pos) {
  assert(step_pos < sboxes.size());
  bool ret_val;
  Mask copyin(GetVerticalMask(step_pos, *in));
  Mask copyout(GetVerticalMask(step_pos, *out));
  ret_val = sboxes[step_pos].Update(copyin, copyout, cache_.get());
  SetVerticalMask(step_pos, *in, copyin);
  SetVerticalMask(step_pos, *out, copyout);
  return ret_val;
}

Mask SyconSboxLayer::GetVerticalMask(unsigned int b,
                                     const StateMaskBase& s) const {
  return Mask(
      { s[4].bitmasks[b], s[3].bitmasks[b], s[2].bitmasks[b], s[1].bitmasks[b],
          s[0].bitmasks[b] });
}

void SyconSboxLayer::SetVerticalMask(unsigned int b, StateMaskBase& s,
                                     const Mask& mask) {


  s[0].bitmasks[b] = mask.bitmasks[4];
  s[1].bitmasks[b] = mask.bitmasks[3];
  s[2].bitmasks[b] = mask.bitmasks[2];
  s[3].bitmasks[b] = mask.bitmasks[1];
  s[4].bitmasks[b] = mask.bitmasks[0];
  BitVector m = ~(1ULL << b);
  s[0].caremask.canbe1 = (s[0].caremask.canbe1 & m)
      | (((mask.caremask.canbe1 >> 4) & 1) << b);
  s[1].caremask.canbe1 = (s[1].caremask.canbe1 & m)
      | (((mask.caremask.canbe1 >> 3) & 1) << b);
  s[2].caremask.canbe1 = (s[2].caremask.canbe1 & m)
      | (((mask.caremask.canbe1 >> 2) & 1) << b);
  s[3].caremask.canbe1 = (s[3].caremask.canbe1 & m)
      | (((mask.caremask.canbe1 >> 1) & 1) << b);
  s[4].caremask.canbe1 = (s[4].caremask.canbe1 & m)
      | (((mask.caremask.canbe1 >> 0) & 1) << b);

  s[0].caremask.care = (s[0].caremask.care & m)
      | (((mask.caremask.care >> 4) & 1) << b);
  s[1].caremask.care = (s[1].caremask.care & m)
      | (((mask.caremask.care >> 3) & 1) << b);
  s[2].caremask.care = (s[2].caremask.care & m)
      | (((mask.caremask.care >> 2) & 1) << b);
  s[3].caremask.care = (s[3].caremask.care & m)
      | (((mask.caremask.care >> 1) & 1) << b);
  s[4].caremask.care = (s[4].caremask.care & m)
      | (((mask.caremask.care >> 0) & 1) << b);
}

