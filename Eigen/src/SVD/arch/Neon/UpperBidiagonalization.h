// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2013-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BIDIAGONALIZATION_H
#define EIGEN_BIDIAGONALIZATION_H

namespace Eigen { 

namespace internal {
// UpperBidiagonalization will probably be replaced by a Bidiagonalization class, don't want to make it stable API.
// At the same time, it's useful to keep for now as it's about the only thing that is testing the BandMatrix class.

template<typename _MatrixType> class UpperBidiagonalization<_MatrixType, Architecture::NEON>
{
  public:

    typedef _MatrixType MatrixType;
    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      ColsAtCompileTimeMinusOne = internal::decrement_size<ColsAtCompileTime>::ret
    };
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef Eigen::Index Index; ///< \deprecated since Eigen 3.3
    typedef Matrix<Scalar, 1, ColsAtCompileTime> RowVectorType;
    typedef Matrix<Scalar, RowsAtCompileTime, 1> ColVectorType;
    typedef BandMatrix<RealScalar, ColsAtCompileTime, ColsAtCompileTime, 1, 0, RowMajor> BidiagonalType;
    typedef Matrix<Scalar, ColsAtCompileTime, 1> DiagVectorType;
    typedef Matrix<Scalar, ColsAtCompileTimeMinusOne, 1> SuperDiagVectorType;
    typedef HouseholderSequence<
              const MatrixType,
              const typename internal::remove_all<typename Diagonal<const MatrixType,0>::ConjugateReturnType>::type
            > HouseholderUSequenceType;
    typedef HouseholderSequence<
              const typename internal::remove_all<typename MatrixType::ConjugateReturnType>::type,
              Diagonal<const MatrixType,1>,
              OnTheRight
            > HouseholderVSequenceType;
    
    /**
    * \brief Default Constructor.
    *
    * The default constructor is useful in cases in which the user intends to
    * perform decompositions via Bidiagonalization::compute(const MatrixType&).
    */
    UpperBidiagonalization() : m_householder(), m_bidiagonal(), m_isInitialized(false) {}

    explicit UpperBidiagonalization(const MatrixType& matrix)
      : m_householder(matrix.rows(), matrix.cols()),
        m_bidiagonal(matrix.cols(), matrix.cols()),
        m_isInitialized(false)
    {
      compute(matrix);
    }
    
    UpperBidiagonalization& compute(const MatrixType& matrix);
    UpperBidiagonalization& computeUnblocked(const MatrixType& matrix);
    
    const MatrixType& householder() const { return m_householder; }
    const BidiagonalType& bidiagonal() const { return m_bidiagonal; }
    
    const HouseholderUSequenceType householderU() const
    {
      eigen_assert(m_isInitialized && "UpperBidiagonalization is not initialized.");
      return HouseholderUSequenceType(m_householder, m_householder.diagonal().conjugate());
    }

    const HouseholderVSequenceType householderV() // const here gives nasty errors and i'm lazy
    {
      eigen_assert(m_isInitialized && "UpperBidiagonalization is not initialized.");
      return HouseholderVSequenceType(m_householder.conjugate(), m_householder.const_derived().template diagonal<1>())
             .setLength(m_householder.cols()-1)
             .setShift(1);
    }
    
  protected:
    MatrixType m_householder;
    BidiagonalType m_bidiagonal;
    bool m_isInitialized;
};

// Standard upper bidiagonalization without fancy optimizations
// This version should be faster for small matrix size
// @Todo:All the three func are defined in householder.h, rewrite these func to NEON.
// makeHouseholderInPlace applyHouseholderOnTheLeft applyHouseholderOnTheRight
template<typename MatrixType> 
void upperbidiagonalization_inplace_unblocked(MatrixType& mat,
                                              typename MatrixType::RealScalar *diagonal,
                                              typename MatrixType::RealScalar *upper_diagonal,
                                              typename MatrixType::Scalar* tempData = 0)
{
  typedef typename MatrixType::Scalar Scalar;

  Index rows = mat.rows();
  Index cols = mat.cols();

  typedef Matrix<Scalar,Dynamic,1,ColMajor,MatrixType::MaxRowsAtCompileTime,1> TempType;
  TempType tempVector;
  if(tempData==0)
  {
    tempVector.resize(rows);
    tempData = tempVector.data();
  }

  for (Index k = 0; /* breaks at k==cols-1 below */ ; ++k)
  {
    Index remainingRows = rows - k;
    Index remainingCols = cols - k - 1;

    // construct left householder transform in-place in A
    mat.col(k).tail(remainingRows)
       .makeHouseholderInPlace(mat.coeffRef(k,k), diagonal[k]);
    // apply householder transform to remaining part of A on the left
    mat.bottomRightCorner(remainingRows, remainingCols)
       .applyHouseholderOnTheLeft(mat.col(k).tail(remainingRows-1), mat.coeff(k,k), tempData);

    if(k == cols-1) break;

    // construct right householder transform in-place in mat
    mat.row(k).tail(remainingCols)
       .makeHouseholderInPlace(mat.coeffRef(k,k+1), upper_diagonal[k]);
    // apply householder transform to remaining part of mat on the left
    mat.bottomRightCorner(remainingRows-1, remainingCols)
       .applyHouseholderOnTheRight(mat.row(k).tail(remainingCols-1).transpose(), mat.coeff(k,k+1), tempData);
  }
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_BIDIAGONALIZATION_H
