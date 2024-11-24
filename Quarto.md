---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Quarto System

So this system is set up with quarto. It should make document creation easy.

Preview documents as follows:

```{code-cell} ipython3
!quarto preview FILE --port $QPORT
```
> [!IMPORTANT] 
> This should be done in a terminal tab as it stays running and updates the output. 

```{code-cell} ipython3
!quarto help
```

