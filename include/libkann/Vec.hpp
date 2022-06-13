#pragma once

#include <stddef.h>
#include <compare>

namespace kann
{
  struct Vec2
  {
  public:
    constexpr Vec2() = default;
    constexpr Vec2(size_t a, size_t b) : a(a), b(b) {}

  public:
    constexpr size_t width()  const { return a; }
    constexpr size_t height() const { return b; }

    constexpr size_t x() const { return a; }
    constexpr size_t y() const { return b; }

  public:
    constexpr friend bool operator==(Vec2 lhs, Vec2 rhs) = default;
    constexpr friend bool operator!=(Vec2 lhs, Vec2 rhs) = default;

  public:
    size_t a, b;
  };

  inline constexpr Vec2 operator+(Vec2 lhs, Vec2 rhs) { return Vec2(lhs.a + rhs.a, lhs.b + rhs.b); }
  inline constexpr Vec2 operator-(Vec2 lhs, Vec2 rhs) { return Vec2(lhs.a - rhs.a, lhs.b - rhs.b); }

  inline constexpr Vec2 operator*(size_t i, Vec2 v) { return Vec2(i * v.a, i * v.b); }
  inline constexpr Vec2 operator*(Vec2 v, size_t i) { return Vec2(v.a * i, v.b * i); }

  inline constexpr Vec2 operator%(Vec2 v, size_t i) { return Vec2(v.a % i, v.b % i); }
  inline constexpr Vec2 operator/(Vec2 v, size_t i) { return Vec2(v.a / i, v.b / i); }
}
