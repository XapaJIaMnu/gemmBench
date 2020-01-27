#pragma once
#include <cstdlib>

// S-byte aligned simple vector. Adapted from kpu's intgemm

namespace alloc {

template <class T> class AlignedVector {
  public:
    explicit AlignedVector(std::size_t size, size_t bytes) {
      // Make sure size is a multiplye of byte:
      size_t size_ = ((size * sizeof(T)) + (bytes -1)) & ~(bytes -1);
      #ifdef __APPLE__
        posix_memalign(reinterpret_cast<void **>(&mem_), bytes, size_);
      #else
        mem_ = static_cast<T*>(aligned_alloc(bytes, size_));
      #endif
    }

    ~AlignedVector() { std::free(mem_); }

    T &operator[](std::size_t offset) { return mem_[offset]; }
    const T &operator[](std::size_t offset) const { return mem_[offset]; }

    T *get() { return mem_; }
    const T *get() const { return mem_; }

  private:
    T *mem_;

    // Deleted.
    AlignedVector(AlignedVector &) = delete;
    AlignedVector &operator=(AlignedVector &) = delete;
};

} // namespace alloc
