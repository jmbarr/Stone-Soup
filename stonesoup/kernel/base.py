import numpy as np

from scipy.special import gamma, kv

from ..base import Property
from ..models.base import Base
from ..types.array import StateVectors, Matrix


class Kernel(Base):
    r"""
    Base class for covariance functions, also known as kernels. In general these map a pair of
    inputs, :math:`\mathbf{x} \in \mathcal{X}` and :math:`\mathbf{x}^{\prime} \in
    \mathcal{X}^{\prime}` to :math:`k(\mathbf{x},\mathbf{x}^{\prime}) \in \mathbb{R}`. See _[1] for
    some background and how these are used in Gaussian processes. In practice this function returns
    the Gram matrix, :math:`K` where :math:`K_{ij} = k(\mathbf{x}_i,\mathbf{x}_j)`.

    Daughter classes of this function must define the :meth:`_kernel_statevector()` method. This
    should specify the kernel as a function of a pair of :class:`~.StateVector` inputs. The parent
    class then takes care of :class:`~.StateVectors` inputs and returns a :class:`~.Matrix` of
    appropriate dimension.

    Reference
    ---------
    _[1] Williams, C.E., Rassmussen, C.K.I. 2006, Gaussian Processes for Machine Learning, the MIT
    Press, Massachusetts Institute of Technology, ISBN 026218253X, www.GaussianProcess.org/gpml

    """
    hyperparameters: dict = Property(doc=" A dictionary of hyperparameters")

    def _kernel_statevector(self, sv1, sv2):
        raise NotImplemented

    def kernel(self, firstin, secondin):
        """
        Function which returns the kernel Gram matrix for the kernel function.

        Parameters
        ----------
        firstin : :class:`~.StateVector`, :class:`~.StateVectors`
         Input StateVector or StateVectors, dimension d x m where d is the dimension of the state
         and m the number of vectors
        secondin : :class:`~.StateVector`, :class:`~.StateVectors`
         Input state vector or state vectors, dimension d x n where d is the dimension of the state
         and n the number of vectors

        Returns
        -------
          :class:`~.Matrix` of dimension m x n.

        """
        out = Matrix(np.zeros([firstin.shape[1], secondin.shape[1]]))
        if type(firstin) is StateVectors:
            if type(secondin) is StateVectors:
                for i, sv1 in enumerate(firstin):
                    for j, sv2 in enumerate(secondin):
                        out[i, j] = self._kernel_statevector(sv1, sv2)
            else:
                for i, sv1 in enumerate(firstin):
                    out[i, 0] = self._kernel_statevector(sv1, secondin)
        else:
            if type(secondin) is StateVectors:
                for j, sv2 in enumerate(secondin):
                    out[0, j] = self._kernel_statevector(firstin, sv2)
            else:
                out[0, 0] = self._kernel_statevector(firstin, secondin)

        return out


class SquaredExponentialKernel(Kernel):
    r"""
    The isotropic squared exponential kernel,

    .. math::

        k(\mathbf{x}, \mathbf{x}^{\prime}) =
        \sigma_f^2 \exp \left( \frac{||\mathbf{x} - \mathbf{x}^{\prime}||^2}{2l^2}\right)

    where :math:`l` is the characteristic length scale and :math:`\sigma_f` is the signal variance.

    Note
    ----
    This kernel is *isotropic* in that it has a single length scale in all dimensions. It is
    possible to construct a more general function via length-scale covariances, though that's not
    done here.


    """
    hyperparameters: dict = Property(doc="Squared exponential hyperparameters should include "
                                         "length scale and signal variance as 'length' and "
                                         "'sigma_f' respectively.")

    def _kernel_statevector(self, sv1, sv2):
        return self.hyperparameters['sigma_f'] ** 2 * \
               np.exp(-0.5 * ((np.linalg.norm(sv1 - sv2)) / self.hyperparameters['length']) ** 2)


class MaternKernel(Kernel):
    r"""
    The Matérn class of kernel,

    .. math::

        k(\mathbf{x}, \mathbf{x}^{\prime}) =
        \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}}{l} ||\mathbf{x} -
        \mathbf{x}^{\prime}|| \right)^{\nu} K_{\nu} \left(\frac{\sqrt{2\nu}}{l}||\mathbf{x} -
        \mathbf{x}^{\prime}||\right)

    for positive parameters, smoothness :math:`\nu` and length scale :math:`l`. :math:`K_{\nu}` is
    the modified Bessel function of the second kind of order :math:`\nu`.

    Note
    ----
    This function isn't well-behaved for large values of :math:`\nu` and small distances (where
    there's essentially a 0 x infinity calculation). Since values of :math:`\nu > 7.2` are rarely
    used because they're barely distinguishable from the squared exponential kernel, the advice is
    to stick to :math:`\nu \leq 7.2` or use the squared exponential if you're after a smooth
    kernel.

    """
    hyperparameters: dict = Property(doc="Parameters for the Matérn kernel should include the "
                                         "length scale and the smoothness parameter as 'length' "
                                         "and 'nu' respectively.")

    def _kernel_statevector(self, sv1, sv2):
        """We have to cope with the fact that the Bessel function at r = 0 is infinite, but the 
        kernel needs to converge to 1"""

        # TODO: This just deals with r = 0, we could expand this to catch nans from the Bessel 
        # TODO: function and return 1. That would help with large nu and small r errors.
        r = np.linalg.norm(sv1 - sv2)
        if r == 0:
            return 1.
        else:
            nu = self.hyperparameters['nu']
            meuc = (np.sqrt(2 * nu) / self.hyperparameters['length']) * r
            gamm = gamma(nu)
            bess = kv(nu, meuc)
            return (2 ** (1 - nu)) / gamm * meuc ** nu * bess
