# libtensor

## Usage
The library simply have one types known as `Tensor<T>` parameterized on the
type it store. To manipuate it, simply use functions you could find in header
files in the include directory.

## Design

### Purity and not using expression template
All functions in this library does not modify its argument but instead return a
new Tensor. This is a property for function is known as purity has the benefit
of making the code much easier to reason about. This has the sided effect of
requiring more memory allocations.

However, memory allocation is cheap provided that you do it
the righy way. If you do:
```
  std::vector<float> data(1000);
```
It would be slow since constructor of vector would value-initialzied its
element. In simpler terms, it would zero-initialize all its element even if you
would later overwrite it anyway. Instead use
```
  std::unique_ptr<float[]> data = std::make_unique_ptr_for_overwrite<float[]>(1000);
```
which has the same effect except the data are left in an uninitialized state,
which is perfect if they are later going to be overwritten anyway.

It would be possible via the use of expression template to eliminate some of
the memory allocations, but the added complexity is not worth it.

### Not supporting general stride
Modern CPU expect data to be stored contiguously to be able to exploit such as
SIMD instruction set. To support general stride while maintaining performance
would be a hard problem, which even if possible to solve would introduce
additional unnecessary complexity.

### Less general operation
A large portion of functions in this library are of the form \*\_inner" and
\*\_outer. One example would be broadcasting and reduction operations. While
it is tempting to make them work with arbitrary axis in the middle, it turns
out that this two use cases is enough for most of what we want to do. It is
expected that by simplifying the interface, optimizing it or moving it to the
GPU for computation would becomes much easier.

This make the library less powerful than other library such as NumPy or
xtensor, with the benefit of being significantly simpler to implement.

## Hacking

### Adding new function on tensor types
Simply look at the implementation of existing functions. You would need to look
at a type called `Buffer<T>` which manages mutable buffer. After creating
`Tensor<T>` using a `Buffer<T>`, the contained data shall not be modified
again.
