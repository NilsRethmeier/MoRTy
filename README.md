# MoRTy ...
is a simple baseline method for *zero-shot domain adaptation* of embeddings that works especially well for *low-resrouce* applications, such as when *little pretraing data is available*. It solves the ...

### Problems
In practice, we have to chose which embedding model (FastText, Glove, TransformerX) is optimal for a task. While most pretraining methods like BERT are optimized for 'high-pretraining-resource' domains, they can not be directly applied to pretraining resource settings and incure substantial training costs. In practice, using a Multi-GPU model to fine tune on a sub 10 MB supervision task can seem counterintuitive and affords preparation and maintanance costs, which limits scalability of future use.  

### MoRTy:
*M*enu *o*f *r*econstructing *t*ransformations *y*ields domain adapted embeddings
