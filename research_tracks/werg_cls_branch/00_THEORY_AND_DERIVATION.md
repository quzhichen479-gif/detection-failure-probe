# WERG v0 — Theory and Derivation

## 1. Starting point: model the water explanation, not a fixed frequency band

Let the accepted P3 classification feature be

\[
F:\Omega\subset\mathbb R^2\to\mathbb R^C.
\]

For a sufficiently small water neighborhood, assume only local smoothness, not a low-frequency identity:

\[
F_w\in C^2(\Omega).
\]

Near a candidate center \(p\), a quadratic local model is

\[
F_w(p+h)\approx b_0+b_x h_x+b_y h_y+b_{xx}h_x^2+b_{xy}h_xh_y+b_{yy}h_y^2.
\]

The center must not participate in its own background estimate. WERG therefore fits this model only on an annular ring \(\mathcal R\) around the candidate.

## 2. Annular quadratic continuation

For normalized local coordinate \(h=(h_x,h_y)\), define

\[
\phi(h)=[1,h_x,h_y,h_x^2,h_xh_y,h_y^2]^T.
\]

For fixed ring samples \(h_j\), construct

\[
X=[\phi(h_1)^T;\ldots;\phi(h_m)^T],\qquad
Y=[F(p+h_1)^T;\ldots;F(p+h_m)^T].
\]

Use a small ridge penalty that leaves the intercept unpenalized:

\[
B=(X^TX+\lambda_pR)^{-1}X^TY,
\quad R=\mathrm{diag}(0,1,1,1,1,1).
\]

Because the ring geometry is fixed, the matrix

\[
K=(X^TX+\lambda_pR)^{-1}X^T
\]

is fixed. Each row of \(K\) can therefore be implemented as a fixed depthwise convolution kernel. No per-location polynomial solve is needed.

The predicted water center is

\[
\hat F_w(p)=b_0.
\]

The center innovation is

\[
r=F(p)-\hat F_w(p).
\]

## 3. Local clutter normalization

Raw \(\|r\|\) is not a valid water anomaly because rough water naturally has larger variation. Let the annular per-channel variance be \(\sigma_c^2\). Use shrinkage

\[
\tilde\sigma_c^2=(1-\eta)\sigma_c^2+\eta\bar\sigma^2+\epsilon,
\]

with

\[
\bar\sigma^2=\frac1C\sum_c\sigma_c^2.
\]

Define the diagonal precision

\[
W=\mathrm{diag}(1/\tilde\sigma_1^2,\ldots,1/\tilde\sigma_C^2).
\]

## 4. Water nuisance tangent projection

The five non-intercept polynomial coefficients define local directions

\[
D=[b_x,b_y,b_{xx},b_{xy},b_{yy}]\in\mathbb R^{C\times5}.
\]

Instead of penalizing the full innovation, remove the component explainable by these local water directions:

\[
\min_a \|r-Da\|_W^2+\lambda_a\|a\|_2^2.
\]

Let

\[
G=D^TWD,\qquad q=D^TWr.
\]

The optimum is

\[
\hat a=(G+\lambda_aI)^{-1}q.
\]

The unexplained energy is

\[
\boxed{S_W=r^TWr-q^T(G+\lambda_aI)^{-1}q.}
\]

With \(A=W^{1/2}D\),

\[
S_W=r^TW^{1/2}\left[I-A(A^TA+\lambda I)^{-1}A^T\right]W^{1/2}r\ge0,
\]

because every singular value contributes \(\sigma_i^2/(\sigma_i^2+\lambda)\in[0,1)\).

For v0, use the conservative degree-of-freedom normalization

\[
T_W=S_W/\max(C-5,1),\qquad z_W=\log(1+T_W).
\]

The exact ridge effective DoF can be audited later; it is not required to test the central hypothesis.

## 5. Transparent-object geometry: why frequency magnitude is insufficient

A simplified transparent/refractive object can be written as

\[
I_o(x)\approx I_w(x+u(x)).
\]

For constant displacement \(u_0\),

\[
\hat I_o(\omega)=e^{i\omega^Tu_0}\hat I_w(\omega),
\]

so

\[
|\hat I_o(\omega)|=|\hat I_w(\omega)|.
\]

Thus an energy-only wavelet/frequency gate can be blind to an important transparent-object case: geometry can change mainly through phase/spatial deformation.

For spatially varying displacement, with

\[
A=I+\nabla u,
\]

the image gradient follows

\[
\nabla I_o\approx A^T\nabla I_w.
\]

The local structure tensor

\[
J=G_\sigma*(\nabla I\nabla I^T)
\]

therefore transforms approximately as

\[
J_o\approx A^TJ_wA.
\]

## 6. Remove scalar photometric gain before measuring geometry

For a local affine intensity change

\[
I'=aI+b,
\]

we have

\[
J'=a^2J.
\]

Direct tensor differences would incorrectly treat a scalar gain as geometry. Regularize and determinant-normalize the SPD tensor:

\[
\bar J=J+\rho\frac{\mathrm{tr}(J)}2I+\epsilon I,
\]

\[
\boxed{Q(J)=\bar J/\sqrt{\det\bar J}.}
\]

Ignoring the tiny numerical floor, \(Q(a^2J)=Q(J)\). Additive brightness is already removed by the gradient.

## 7. Center-ring deformation distance

Let \(Q_c\) be the candidate-center normalized tensor and \(Q_r\) the annular background normalized tensor. Use the affine-invariant SPD distance

\[
D_G^2=\left\|\log(Q_r^{-1/2}Q_cQ_r^{-1/2})\right\|_F^2.
\]

For determinant-normalized 2x2 tensors, the relative eigenvalues are \(\lambda,1/\lambda\). Therefore

\[
D_G^2=2(\log\lambda)^2.
\]

Because

\[
\frac12\mathrm{tr}(Q_r^{-1}Q_c)=\cosh(\log\lambda),
\]

WERG computes the equivalent scalar form

\[
\boxed{D_G^2=2\,\mathrm{acosh}^2\left(\tfrac12\mathrm{tr}(Q_r^{-1}Q_c)\right).}
\]

Near identity, with \(x=z-1\),

\[
D_G^2\approx4x-\frac23x^2,
\]

which avoids unstable differentiation/evaluation of `acosh` extremely close to 1.

A low-gradient confidence multiplies \(D_G^2\) so nearly textureless regions do not obtain arbitrary orientation-shape scores from numerical noise. The final scalar is

\[
z_G=\log(1+D_G^2_{conf}).
\]

## 8. Small-deformation interpretation

Let \(A=I+E\), \(E=\nabla u\), and consider an approximately isotropic local background tensor. To first order,

\[
J_o\approx\lambda[I+E+E^T].
\]

After determinant normalization, the isotropic expansion term is removed, leaving approximately

\[
Q_o\approx I+2\,\mathrm{dev}(\mathrm{Sym}(E)).
\]

Therefore

\[
\boxed{D_G^2\approx4\|\mathrm{dev}(\mathrm{Sym}(\nabla u))\|_F^2.}
\]

So the geometry statistic is aimed at anisotropic refractive strain, not generic high frequency or gradient amplitude.

## 9. Classification correction

WERG does not use a hard threshold. Given accepted semantic pre-sigmoid P3 logits \(\ell_{c,p}\), define

\[
\phi=[1,z_W,z_G,z_W^2,z_Wz_G,z_G^2]^T.
\]

With only six trainable coefficients \(w\),

\[
s=w^T\phi,
\]

and a bounded shared correction

\[
\boxed{\Delta\ell=\gamma\tanh(s/\gamma),\qquad \ell'_{c,p}=\ell_{c,p}+\Delta\ell_p.}
\]

`w=0` at initialization, hence the inserted module is exactly the accepted parent at step 0. The correction is shared across classes because it represents physical object-vs-water evidence, not class identity.

## 10. Explicit failure conditions

WERG cannot create information that is absent from a single RGB frame. It may fail for:

- a transparent object on a nearly uniform background with negligible reflection and near-constant displacement;
- saturated/very sharp glitter where the photometric field itself has a strong spatial derivative;
- foam that is both semantically and geometrically object-like;
- a P3 feature field that does not exhibit the assumed local smooth/nuisance structure;
- transparent-object deformation already destroyed by optical blur, compression or downsampling.

These are not implementation bugs. They are pre-registered falsification cases.
