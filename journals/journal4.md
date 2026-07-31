We're so close to start development! After studying many attention models and transformer block variants, our next step is to study Positional encoding.
The SOTA thing nearly everybody does (even though each model varies the use of that) is ROPE: **Rotary Positional Encodings**
This technique rotates query and key vectors according to their position in the tokens sequence. 

At position m, RoPE applies a rotation: R(mθ)=[ cos(mθ) sin(mθ) ​ −sin(mθ) cos(mθ) ​ ]. The rotated query is: q m RoPE ​ =R(mθ)q. The key at position n is rotated similarly: k n RoPE ​ =R(nθ)k.