<h2> Randomized Extended Kaczmarz (C++) <img src="https://travis-ci.org/zouzias/REK-CPP.svg?branch=master"/></h2>
<h3> Overview </h3>
The current project contains a CPP implementation of the algorithm described in the research paper [1]. The Randomized Extended Kaczmarz algorithm is a randomized algorithm for solving least-squares/linear regression problems.

<br>
[<b>Randomized Extended Kaczmarz for Solving Least-Squares.</b>](http://dx.doi.org/10.1137/120889897)
<br>
SIAM. J. Matrix Anal. & Appl., 34(2), 773–793. (21 pages) 
<br>
Authors: [Anastasios Zouzias](https://github.com/zouzias) and Nikolaos Freris 
<br>
<h3>
Installation
</h3>

Clone the project. Type type

<code>make && make test</code>
<br>
<code>./main</code>

The above code runs a simple instance of least-squares for a gaussian random matrix A and gaussian vector b. See 'main.c' for more details.
<h4>
Bugs
</h4>
Please report bugs by opening a new [issue](https://github.com/zouzias/REK-CPP/issues/new).

<h4>
Implementation Details
</h4>
REK-BLAS is an implementation of REK with two additional technical features. First, REK-BLAS utilizes level-1 BLAS routines for 
all operations of REK and second REK-BLAS additionally stores explicitly the transpose of A for more efficiently 
memory access of both the rows and columns of A using BLAS (see the above paper for more details). 

The sampling operations of REK are implemented using the so-called ``alias method'' for generating samples 
from any given discrete distribution [Vos91]. In particular, the alias method, assuming access 
to a uniform random variable on [0,1] in constant time and linear time preprocessing, generates one sample
of a given distribution in constant time. We use an implementation of W. D. Smith.

<h4>
Credits and acknowledgments
</h4>

Credits go to Warren D. Smith for implementing the aliasing method [Vos91] in C.
<br><br>
[Vos91] M. D. Vose. A Linear Algorithm for Generating Random Numbers with a given Distribution. 
<br>
IEEE Trans. Softw. Eng., 17(9):972–975, September 1991.
