# vugrad

```vugrad``` is a miniature autodiff system. Think Pytorch or Tensorflow, but designed to be easy to understand (and not suitable for production).

There are many packages like this (see below for a list). Here are vugrads specifics:
 * Built on numpy, which is the only requirement.
 * Tensor-valued: all operations are on numpy arrays.
 * Eager execution: computation graphs are built on the fly.
 
```vugrad``` was built for the [Deep Learning course](http://dlvu.github.io) at the Vrije Universiteit Amsterdam. 

## Similar packages 

The first package to do something like this was probably [micrograd](https://github.com/karpathy/micrograd) By Andrej Karpathy, followed quickly by [minigrad](https://github.com/kennysong/minigrad) by Kenny Song. These are both scalar-valued, which mean they don't illustrate how tensors are handled.

On the other end of the spectrum, we have [minitorch](https://minitorch.github.io/) which is a full reimplementation of the Pytorch API in educational code. That means they don't defer to numpy for the tensor implementation, like we do, but they build that from scratch as well. 


