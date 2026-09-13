#ifndef NLHE_BUCKETING_HAND_ISOMORPHISM_H
#define NLHE_BUCKETING_HAND_ISOMORPHISM_H

// Thin C++ adapter for Kevin Waugh's hand-isomorphism library.
//
// Two reasons it cannot be included from C++ directly:
//   1. hand_index.h declares its API returning the C keyword `_Bool` and has no
//      `extern "C"` guard, so C++ rejects it. We map `_Bool` -> `bool` for the
//      duration of the include and give the declarations C linkage. The library
//      sources (hand_index.c, deck.c) are compiled as C by CMake.
//   2. deck.h (pulled in transitively) #defines the very generic object-like
//      macros SUITS / RANKS / CARDS. We #undef them on the way out so they
//      cannot leak into and collide with other code in the translation unit.
//
// Nothing in the upstream checkout is modified; the shim lives entirely here.

#include <cstdint>
#include <cstddef>

extern "C" {
#define _Bool bool
#include "hand_index.h"
#undef _Bool
}

#undef SUITS
#undef RANKS
#undef CARDS

#endif  // NLHE_BUCKETING_HAND_ISOMORPHISM_H
