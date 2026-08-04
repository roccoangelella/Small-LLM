Today we hopefully define the last training stuff and try making the whole training begin. 
We successfully defined the final model architecture and verified that (at least for now) we can safely train it on a T4 GPU. First experiments will be slow but free, and we can iterate on them until being safe enough to start using the 6000 PRO.

The final stuf to define is:
1. DataLoader (we'll mainly delegate this part but it'll still be interesting to explore)
2. Optimizer
3. Scheduler
4. Weights initialization (we'll use Gaussian since xavier is broken, at least for now.)

I'd start from Optimizer. The SOTA optimizers, used both in Kimi K3 and Deepseek v4 are:
- **MUON** (MomentUm Orthogonalized Newton-Schulz): it aims at "not trusting" the dominating gradient update directions, smoothing the direction imbalance caused by momentum. Muon "redistributes" the magnitude of gradient updates obtained through momentum: first it computes the classic momentum update matrix, then it takes its singular value decomposition, denoted SVD. 

What SVD does (brief linear algebra review), is expressing a matrix M as the product of the matrices M=U*\sigma*V^T., where U and V describe a rotation, and sigma represents a shrink/stretch. the definition of SVD i give myself is tat, provided a transformation matrix A, we use SVD to split the overall transformation in "Two rotations and one shrink/stretch".

So, in SVD, U,V represent which directions matter the most in our transformation, and \sigma_caps describes **how strongly each direction is stretched/shrunk**

So, after taking the SVD of the Momentum matrix, **Muon replaces \sigma_caps values with approximately the identity matrix**, therefore our SVD becomes roughly UV^T. Therefore, the directions that exist in each matrix update obtain roughly the same importance.

A final nuance: Muon couples all attention heads together: heads with larger momentum can dominate the orthogonalization. Therefore, SOTA models like Kimi K3 apply muon separately per each head.

Muon is usually not applied to every parameter in an LLM. Instead, Muon and AdamW can optimize different parameters within the same layer.

Muon generally handles two-dimensional weight matrices that perform ordinary transformations between feature spaces, such as the (W_Q), (W_K), (W_V), and output-projection matrices in multi-head attention, or the gate, up, and down projection matrices in SwiGLU.

AdamW handles the remaining parameters, including embeddings, normalization scales, biases, scalar or vector parameters, and specialized parameters that represent recurrence dynamics, temporal filters, or other structured operations.

Therefore, the division is based primarily on each parameter’s mathematical role, not simply on which layer it belongs to.
The sota for fixing the learning rate of this architecture is:
AdamW LR = scheduled LR
Muon LR  = scheduled LR × Muon multiplier
- **SOAP** (Shampoo with Adam in Preconditioner's eigenbasis): it combines ideas from Shampoo and Adam optimimzers. It keeps a moving average of gradients for every weight, denoted G, kept "stored" in two different ways, L=GG^T and R=G^TG. L describes the relationship among rows, and R among columns, highlighting which combinations of rows and columns tend to move together. 

Soap performs **eigendecomposition**: given a square matrix A, we rewrite it as A=QΛQ^-1. Q is the matrix whose columns are eigenvectors of A, lambda is the diagonal matrix of eigenvalues. For a symmetric matrix (L,R are symmetric), A=QΛQ^T. Therefore we get L=QL​ΛL​QL⊤;​R=QR​ΛR​QR⊤​. By definition, QL and QR will contain the most useful directions!

Then, at each step, SOAP rotates the moving avg of the gradients in this coordinate system: G'=Q^TL*GQR. This rotated Gradient matrix features directions aligned with coordinate axes (won't go deeper on the meaning of this, i'm tired lol). Done that, we apply Adam on the rotated space, using the classic two moments update, and finally rotating back into the original matrix coordinates by A=QL​A′QR⊤​.

Finally, Wnew​=W−ηA.