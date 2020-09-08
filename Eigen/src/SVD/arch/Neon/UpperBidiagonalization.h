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



} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_BIDIAGONALIZATION_H
