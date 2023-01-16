# hp-MG methods for the H(div)-HDG for the incompressible flow problems
Paper link: TO BE FILLED.

A proceeding work of our previous study on [optimal geometric h-MG for HDG-P0 for the reaction-diffusion and the generalized Stokes equations](https://arxiv.org/abs/2208.14418).

Augmented Lagrangian (AL) Uzawa iteration method is used to solve the mixed H(div)-HDG scheme for the generalized Stokes and NS equations.

We firsly proved the equivalence between the lowest-order mixed H(div)-HDG scheme and the well-established Crouzeix-Raviart elements for the generalized Stokes equations. Then we naturally proposed a geometric *h*-MG method robust to the mesh size *h* and AL parameter *lambda* of the lowest-order scheme for the generalized Stokes equations, basded on the Ph.D. thesis of Joachim Schöberl (1999).
Then the geometric h-MG of the lowest-order scheme is used as the coarse solver in a multiplicative p-MG of the higher-order H(div)-HDG scheme for the generalized Stokes equations, with robustness with respect to polynomial order demonstrated in the numerical experiments.

The developed hp-MG method for the generalized Stokes equatsions is then extended to the NS equations with an upwind convection term, and numerical experiments are performed.

## Some Hackers to NGSolve Source Files:
1. Make the public parameter "print" of class "FESpace" usable in Python coding.
```C++
/// /comp/fespace.hpp, line 400
docu.Arg("print") = "bool = False\n"
      "  Write additional information to testout file. \n"
      "  This file must be set by ngsolve.SetTestoutFile. Use \n"
      "  ngsolve.SetNumThreads(1) for serial output";
```

2. Added Python interface to the public method of class "FESpace", to get edge/vertex-patched blocks of global/local DOFs of FESpaces in Python to accelerate constructing block GS smoothers.
```C++
/// /comp/python_comp.cpp, line 811
.def("CreateSmoothBlocks", 
    [] (shared_ptr<FESpace> self, bool vertex, bool globalDofs)
    {
    Flags flags;
    flags.SetFlag("eliminate_internal", globalDofs);
    if (vertex)
        flags.SetFlag("blocktype", "vertexpatch");
    else
        flags.SetFlag("blocktype", "edgepatch");

    std::shared_ptr<Table<int>> tablePtr = self -> CreateSmoothingBlocks(flags);
    Table<int> table = move(*(tablePtr.get()));
    std::vector<std::vector<int>> vec;
    for(auto subArr : table)
        {
        if(subArr.Size() == 0)
            continue;
        std::vector<int> tmpv;
        for(auto i : subArr.Range())
            tmpv.push_back(subArr[i]);
        vec.push_back(tmpv);
        }
    return vec;
    },
    py::arg("vertex"), py::arg("globalDofs"),
    "Create vertex/edge-patched blocks of global/local DOFs")
```

3. Added to class "FESpace" a public method "CreateFacetBlocks" and its Python interface, to get facet blocks of global/local DOFs of FESpaces in Python to accelerate mass matrix inverse in 3D cases.
```C++
/// /comp/python_comp.cpp, line 839
.def("CreateFacetBlocks", 
    [] (shared_ptr<FESpace> self, bool globalDofs)
    {

    std::shared_ptr<Table<int>> tablePtr = self -> CreateFacetBlocks(globalDofs);
    Table<int> table = move(*(tablePtr.get()));
    std::vector<std::vector<int>> vec;
    for(auto subArr : table)
        {
        if(subArr.Size() == 0)
            continue;
        std::vector<int> tmpv;
        for(auto i : subArr.Range())
            tmpv.push_back(subArr[i]);
        vec.push_back(tmpv);
        }
    return vec;
    },
    py::arg("globalDofs"),
    "Create facet blocks of global/local DOFs")
```

```C++
/// /comp/fespace.hpp, line 581
shared_ptr<Table<int>> CreateFacetBlocks (const bool globalDofs=true) const;
```

```C++
/// /comp/fespace.cpp line 1584
shared_ptr<Table<int>> FESpace :: CreateFacetBlocks (const bool globalDofs) const
  {
    auto freedofs = GetFreeDofs(globalDofs);
    FilteredTableCreator creator(freedofs.get());
  
    Array<DofId> dofs;
    for ( ; !creator.Done(); creator++)
        for (size_t i : Range(ma->GetNFaces()))        
          {
            // Ng_Node<2> face = ma->GetNode<2> (i);
            
            GetDofNrs (NodeId(NT_FACE, i), dofs);
            for (auto d : dofs)
              if (IsRegularDof(d))
                  creator.Add (i, d);
          }
    Table<int> table = creator.MoveTable();
    if (print)
      *testout << "facet blocks = " << endl << table << endl;
    return make_shared<Table<int>> (move(table));
  }
```

4. Overload the virtual public methods "GetEdgeDofNrs" and "GetFaceDofNrs" of the child class "NonconformingFESpace" from the base class "FESpace", to get usable smoothing bloks of Crouzeix-Raviart spaces.
```C++
/// /comp/fespace.hpp, line 1070
virtual void GetEdgeDofNrs (int ednr, Array<DofId> & dnums) const override;
virtual void GetFaceDofNrs (int fanr, Array<DofId> & dnums) const override;

```

```C++
/// /comp/fespace.cpp, line 2735
void NonconformingFESpace :: GetEdgeDofNrs (int ednr, Array<DofId> & dnums) const
  {
    dnums.SetSize0();
    // for 2D only
    if (ma->GetDimension()==2)
      {
        auto freedofs = GetFreeDofs(false);
        if (freedofs -> Test(ednr))
          {
            dnums.SetSize(1);
            dnums[0] = ednr;
          }
      }
  }
void NonconformingFESpace :: GetFaceDofNrs (int fanr, Array<DofId> & dnums) const
  {
    dnums.SetSize0();
    // for 3D only
    if (ma->GetDimension()==3)
      {
        auto freedofs = GetFreeDofs(false);
        if (freedofs -> Test(fanr))
          {
            dnums.SetSize(1);
            dnums[0] = fanr;
          }
      }
  }
```

