// The XPL CUDA instrumenter

#include <iostream>
#include <memory>

// Mandatory include headers
#include "rose.h"
#include "plugin.h"

// optional headers
#include "sageGeneric.h"
#include "sageBuilder.h"
#include "sageUtility.h"
#include "sageInterface.h"

#include "transformations.h"
#include "xpl-pragmas.h"

namespace su = SageUtil;
namespace sb = SageBuilder;
namespace si = SageInterface;

namespace
{
  /// \brief  applies the function object fun to each pair in the ranges [first1, last1)
  ///         and [first2, first2 + (last1-first1) ).
  /// \return a copy of the function object after it has completed the traversal.
  template <class InputIterator, class InputIterator2, class BinaryFunction>
  inline
  BinaryFunction
  zip(InputIterator first, InputIterator last, InputIterator2 first2, BinaryFunction fun)
  {
    while (first != last)
    {
      fun(*first, *first2);
      ++first; ++first2;
    }

    return fun;
  }

  inline auto display()  -> std::ostream& { return std::cerr; }
  inline auto logInfo()  -> std::ostream& { return std::cerr << "[Info:    ]"; }
  inline auto logWarn()  -> std::ostream& { return std::cerr << "[Warning: ]"; }
  inline auto logError() -> std::ostream& { return std::cerr << "[Error:   ]"; }
}


namespace tracer
{
namespace
{
template <class SageDecl>
bool setKeyDeclIf(bool cond, SageDecl& dcl, SageDecl*& var)
{
  if (!cond) return false;

  var = &su::keyDecl(dcl);
  ROSE_ASSERT(var);
  return true;
}


struct InstrumentationGuide : AstAttribute
{
    enum what { none, skip, read, readWrite };

  private:
    static constexpr const char* KEY = "RAJATRC";

    explicit
    InstrumentationGuide(what instruction)
    : info(instruction)
    {}

    AstAttribute::OwnershipPolicy getOwnershipPolicy() const ROSE_OVERRIDE
    {
      return CONTAINER_OWNERSHIP;
    }

    InstrumentationGuide* copy () const ROSE_OVERRIDE
    {
      return new InstrumentationGuide(*this);
    }

    InstrumentationGuide* constructor () const ROSE_OVERRIDE
    {
      return new InstrumentationGuide(none);
    }

    std::string attribute_class_name () const ROSE_OVERRIDE
    {
      return KEY;
    }

    std::string toString()
    {
      if (info == none) return "none";
      if (info == skip) return "skip";
      if (info == read) return "read";

      return "readWrite";
    }

  public:
    static
    void note(SgNode& n, what instruction)
    {
      n.setAttribute(KEY, new InstrumentationGuide(instruction));
    }

    static
    what note(SgNode& n)
    {
      if (!n.attributeExists(KEY))
        return none;

      return static_cast<InstrumentationGuide*>(n.getAttribute(KEY))->info;
    }

  private:
    what info;
};
} // end anonymous namespace

  struct IsRefType : sg::DispatchHandler<bool>
  {
    typedef sg::DispatchHandler<bool> base;
    
    IsRefType()
    : base(false)
    {}
    
    void handle(SgNode& n)              { SG_UNEXPECTED_NODE(n); }
    void handle(SgType&)                {}
    void handle(SgReferenceType&)       { res = true; }
    void handle(SgRvalueReferenceType&) { res = true; }
  };


  bool isRefType(SgType& ty)
  {
    // return si::isReferenceType(&su::skipTypeModifier(ty));
    return sg::dispatch(IsRefType(), &su::skipTypes(ty, su::modifiers | su::aliases));
  }

  bool hasRefType(SgExpression& n)
  {
    return isRefType(su::type(n));
  }

  bool isCudaCoordinate(SgExpression& n)
  {
    SgVarRefExp* var = isSgVarRefExp(&n);
    if (!var) return false;

    SgName       varname = su::nameOf(*var);

    return (  varname == "blockDim"
           || varname == "blockIdx"
           || varname == "gridIdx"
           || varname == "threadIdx"
           );
  }


  struct ConstPrimitiveType : sg::DispatchHandler<bool>
  {
    typedef sg::DispatchHandler<bool> base;

    ConstPrimitiveType()
    : base(false), cst(false)
    {}

    bool recurse(SgNode* ty) { return sg::dispatch(ConstPrimitiveType(*this), ty); }

    void handle(SgNode& n)          { SG_UNEXPECTED_NODE(n); }

    void handle(SgType& n)          { /* default */ }

    void handle(SgModifierType& n)
    {
      cst = n.get_typeModifier().get_constVolatileModifier().isConst();
      res = recurse(n.get_base_type());
    }

    void handle(SgTypeInt& n)       { res = cst; }
    void handle(SgTypeLongLong& n)  { res = cst; }
    /* \todo handle other primitive types */

    void handle(SgTypedefType& n)   { res = recurse(n.get_base_type()); }
    void handle(SgReferenceType& n) { res = recurse(n.get_base_type()); }

    bool cst;
  };

  bool hasStaticConstPrimitiveType(SgVarRefExp& n)
  {
    SgInitializedName&           init = su::keyDecl(n);
    SgVariableDeclaration*       decl = sg::ancestor<SgVariableDeclaration>(&init);

    if (decl == nullptr) return false;

    const SgDeclarationModifier& modf = decl->get_declarationModifier();

    if (!modf.get_storageModifier().isStatic()) return false;

    return sg::dispatch(ConstPrimitiveType(), n.get_type());
  }

  void annotate(SgExpression& n, InstrumentationGuide::what attr)
  {
    InstrumentationGuide::note(n, attr);
    
    if (SgCastExp* cast = isSgCastExp(&n))
    {
      // \todo do we need to check whether n was compiler generated?
      annotate(SG_DEREF(cast->get_operand()), attr);
    }
  }

  InstrumentationGuide::what
  note(SgExpression& n)
  {
    return InstrumentationGuide::note(n);
  }

  struct FunctionTargetHandler : sg::DispatchHandler<bool>
  {
    void handle(SgNode& n)          { SG_UNEXPECTED_NODE(n); }

    void handle(SgExpression& n)    { res = false; }

    void handle(SgNonrealRefExp&)   { res = true;  }

    void handle(SgFunctionRefExp& n)
    {
      static const std::set<std::string> ignoreList = { "__assert_fail" };
                                    
      SgFunctionSymbol* sym = n.get_symbol();
      
      res = (  sym != nullptr
            && ignoreList.find(sym->get_name()) != ignoreList.end()
            );
    }
  };

  bool skipFunctionTarget(SgExpression* n)
  {
    return sg::dispatch(FunctionTargetHandler(), n);
  }  

  struct CloningBackInserter : std::iterator< std::output_iterator_tag, SgExpression*, void, void, void >
  {
    explicit
    CloningBackInserter(SgExprListExp& container)
    : lst(container)
    {}

    void push_back(SgExpression* exp)
    {
      ROSE_ASSERT(exp);

      lst.append_expression(si::deepCopy(exp));
    }

    void push_back(SgExpression* exp, int defaultVal)
    {
      if (exp) { push_back(exp); return; }

      lst.append_expression(sb::buildIntVal(defaultVal));
    }

    CloningBackInserter& operator=(SgExpression* el)
    {
      push_back(el);
      return *this;
    }

    CloningBackInserter& operator*()
    {
      return *this;
    }

    CloningBackInserter& operator++()
    {
      return *this;
    }

    CloningBackInserter& operator++(int)
    {
      return *this;
    }

    SgExprListExp& lst;
  };

  struct FunctionBaseName : sg::DispatchHandler<std::string>
  {
    void handle(SgNode& n)                { SG_UNEXPECTED_NODE(n); }
    void handle(SgFunctionDeclaration& n) { res = n.get_name(); }

    void handle(SgTemplateInstantiationFunctionDecl& n)
    {
      SgTemplateFunctionDeclaration& dcl = SG_DEREF(n.get_templateDeclaration());

      res = dcl.get_name();
    }
  };
  
  //
  struct ExcludeTemplates
  {
    void handle(SgTemplateClassDeclaration&)          {}
    void handle(SgTemplateClassDefinition&)           {}
    //~ void handle(SgTemplateFunctionDeclaration&)       {} // we cannot skip tracer functions
    void handle(SgTemplateMemberFunctionDeclaration&) {}
    void handle(SgTemplateFunctionDefinition&)        {}
    void handle(SgTemplateTypedefDeclaration&)        {}
    void handle(SgTemplateVariableDeclaration&)       {}
  };
  
  struct SkipRefArgMarker
  {
    void markRefArgs(SgExpression& arg, SgType& ty)
    {
      const bool hasRefT = isRefType(ty);
      
      if (!hasRefT) return;
      
      annotate(arg, InstrumentationGuide::skip);
    }
    
    void operator()(SgExpression* arg, SgType* ty)
    {
      markRefArgs(SG_DEREF(arg), SG_DEREF(ty));
    }
    
    bool err = false;
  };
  
  struct CallArgTypes : sg::DispatchHandler<SgTypePtrList*>
  {
    void recurse(SgType* ty) 
    {
      res = sg::dispatch(CallArgTypes(), ty);
    }
    
    void handle(SgNode& n)         { SG_UNEXPECTED_NODE(n); } 
    void handle(SgType& n)         { SG_UNEXPECTED_NODE(n); } 
    
    void handle(SgPointerType& n)
    { 
      recurse(n.get_base_type()); 
    }
     
    void handle(SgFunctionType& n) { res = &n.get_arguments(); } 
  };
  
  SgTypePtrList& 
  callArgTypes(SgType* ty)
  {
    return *sg::dispatch(CallArgTypes(), ty);
  }


  struct Explorer : ExcludeTemplates
  {
      typedef std::map<std::string, SgFunctionDeclaration*> FunctionReplacementMap;
      typedef std::vector<instrument::AnyTransform>         InstrumentationStore;
      typedef std::vector<std::string>                      ScopeList;
      
      //
      // constructors

      Explorer()                           = default;
      Explorer(Explorer&&)                 = default;
      Explorer& operator=(Explorer&&)      = default;
      
      Explorer(const Explorer&)            = delete;
      Explorer& operator=(const Explorer&) = delete;

      //
      // auxiliary functions

      bool tracerFunction(SgFunctionDeclaration& n)
      {
        SgName name = n.get_name();

        return matchFirstOf
               || setKeyDeclIf(name == "traceR",  n, trRead)
               || setKeyDeclIf(name == "traceW",  n, trWrite)
               || setKeyDeclIf(name == "traceRW", n, trReadWrite)
               ;
      }

      bool skipInstrumentation(const SgDeclarationStatement& n) const
      {
        // delay descending into function bodies until the main
        //   instrumentation functions have been identified.
        return !(trReadWrite && trWrite && trRead);
      }

      template <class Action>
      void record(Action action)
      {
        if ((replacements.size() % 1024) == 0)
          std::cerr << '\r' << replacements.size() << "  ";

        replacements.emplace_back(std::move(action));
      }
      
      template <class Action>
      void recordIf(Action action)
      {
        if (!action.valid()) return;
        
        record(std::move(action));
      }


      void functionReplacement(XPLPragma::FunctionReplacement);

      void handleExpr(SgExpression& n);
      void instrumentExpr(SgExpression& n);
      
      /// returns true if \ref n is part of an excluded scope
      bool excludedScope(SgExpression& n) const;

      //
      // handlers over AST
      
      using ExcludeTemplates::handle;

      void handle(SgNode& n)                 { SG_UNEXPECTED_NODE(n); }

      // support nodes
      void handle(SgProject& n)              { descend(n); }
      void handle(SgFileList& n)             { descend(n); }
      void handle(SgSourceFile& n)           { descend(n); }
      void handle(SgInitializedName& n)      { descend(n); }

      void handle(SgFunctionParameterList&)  { /* skip; */ }

      void handle(SgStatement& n)            
      { 
        //~ std::cerr << "S " << typeid(n).name() << std::endl;
        descend(n); 
      }

      void handle(SgFunctionDefinition& n)
      {
        descend(n);
      }

      void handle(SgReturnStmt& n)
      {
        SgFunctionDeclaration& func = sg::ancestor<SgFunctionDeclaration>(n);
        
        if (isRefType(SG_DEREF(SG_DEREF(func.get_type()).get_return_type())))
          annotate(SG_DEREF(n.get_expression()), InstrumentationGuide::skip);

        descend(n);
      }

      void handle(SgDeclarationStatement& n)
      {
        matchFirstOf
        || skipInstrumentation(n)
        || defaults(&Explorer::descend, std::ref(n))
        ;
      }

      void handle(SgFunctionDeclaration& n)
      {
        matchFirstOf
        || tracerFunction(n)       /* needs to be ordered before skipInstrumentation. */
        || skipInstrumentation(n)
        || defaults(&Explorer::descend, std::ref(n))
        ;
      }

      void handle(SgPragmaDeclaration& n)
      {
        XPLPragma xplpragma(n, SG_DEREF(n.get_pragma()));

        switch (xplpragma.which())
        {
          case XPLPragma::noxpl:
            break;

          case XPLPragma::diagnostic:
            recordIf(xplpragma.action());
            break;
          
          case XPLPragma::ignore_diagnostic:
            xplpragma.updateIgnoreList();
            break;
            
          case XPLPragma::replace:
            functionReplacement(xplpragma.functionReplacement());
            break;

          case XPLPragma::sync_kernel:
            cudaKernelsRequireSynchronize = true;
            break;

          case XPLPragma::exclude_scopes:
            xplpragma.updateExcludedScopes(excludedScopes);
            break;
            
          case XPLPragma::xplon:
            xplactive = true;
            break;
            
          case XPLPragma::xploff:
            xplactive = false;
            break;

          case XPLPragma::undefined: /* fall through */;
          case XPLPragma::unknown:   /* fall through */;
          default:
            throw std::runtime_error( std::string("unknown XPL pragma: ")
                                    + SG_DEREF(n.get_pragma()).get_pragma()
                                    );
        }
      }

      void handle(SgLambdaCapture&)          { /* skip; */ }
      void handle(SgLambdaCaptureList&)      { /* skip; */ }
      void handle(SgCudaKernelExecConfig&)   { /* skip; */ }
      
      void handle(SgCastExp& n)              { descend(n); }

      void handle(SgExpression& n)           { handleExpr(n); }

      void handle(SgExprListExp& n)          { descend(n); }
      void handle(SgSizeOfOp&)               { /* skip sizeof (...) */ }

      void handle(SgConstructorInitializer& n)
      {
        SgMemberFunctionDeclaration* dcl = n.get_declaration();

        if (!dcl || !dcl->get_type()) return;

        descend(n);
      }

      void handleMemberAccess(SgBinaryOp& n);

      void handle(SgDotExp& n)      { handleMemberAccess(n); }
      void handle(SgArrowExp& n)    { handleMemberAccess(n); }

      void handleUnaryRW(SgUnaryOp& n)
      {
        SgExpression& oper = instrument::operand(n);

        annotate(oper, InstrumentationGuide::readWrite);
        handleExpr(n);
      }

      void handle(SgPlusPlusOp& n)   { handleUnaryRW(n); }
      void handle(SgMinusMinusOp& n) { handleUnaryRW(n); }

      void handle(SgCompoundAssignOp& n)
      {
        SgExpression& oper = instrument::lhs(n);

        annotate(oper, InstrumentationGuide::readWrite);
        handleExpr(n);
      }

      void handle(SgPntrArrRefExp& n);

      void handle(SgAddressOfOp& n)
      {
        SgExpression& oper = instrument::operand(n);

        annotate(oper, InstrumentationGuide::skip);
        // do not instrument address taken
        descend(n);
      }

      void handle(SgFunctionRefExp& n)
      {
        SgFunctionDeclaration*           fundecl = n.getAssociatedFunctionDeclaration();
        if (!fundecl) return;

        std::string                      funname = sg::dispatch(FunctionBaseName(), fundecl);
        FunctionReplacementMap::iterator pos = replmap.find(funname);

        if (pos == replmap.end()) return;

        // std::cerr << funname << " / " << n.get_parent()->unparseToString() << std::endl;
        record( instrument::repl(n, mkFunctionTarget(SG_DEREF(pos->second))) );
      }

      void handle(SgMemberFunctionRefExp& n) { /* skip; */ }

      void handle(SgAssignInitializer& n)
      {
        ROSE_ASSERT(!isSgCudaKernelExecConfig(&instrument::initializer(n)));
        
        // skip traversal of operand (mixed up lvalue handling)
        annotate(instrument::initializer(n), InstrumentationGuide::read);
        descend(n);
      }

      void handle(SgVarRefExp& n)
      {
        // \todo instrument variables with CUDA storage properties

        matchFirstOf
        || !hasRefType(n)
        || hasStaticConstPrimitiveType(n)
        || defaults(&Explorer::handleExpr, std::ref(n))
        ;
      }

      void handle(SgThisExp&) { /* skip */ }

      void handle(SgFunctionCallExp& n)
      {
        typedef SgExpressionPtrList::iterator expr_iterator;
        
        SgExpression&        tgtexp = SG_DEREF(n.get_function()); 
        
        // skip certain function calls
        if (skipFunctionTarget(&tgtexp)) return;
        
        SgExpressionPtrList& args  = su::arglist(n);
        SgTypePtrList&       argty = callArgTypes(tgtexp.get_type());
        expr_iterator        argaa = args.begin();
        size_t               len   = std::min(args.size(), argty.size());

        zip(argaa, argaa+len, argty.begin(), SkipRefArgMarker());
        
        // Only reference types can be Lvalues, so we can skip other calls
        if (!hasRefType(n))
        {
          descend(n);
          return;
        }
        
        handleExpr(n);
      }


      void synchronizeKernelIfNeeded(SgCudaKernelCallExp& n)
      {
        if (!cudaKernelsRequireSynchronize) return;

        SgStatement&      node   = sg::ancestor<SgStatement>(n);
        ROSE_ASSERT(&node == n.get_parent());
        ROSE_ASSERT(!isSgScopeStatement(&node));

        SgScopeStatement& scope  = sg::ancestor<SgScopeStatement>(n);
        SgExpression&     callee = mkFunctionTarget("cudaDeviceSynchronize", scope);
        SgExprListExp&    args   = mkArgList();
        SgCallExpression& call   = mkFunctionCallExp(callee, args);
        SgStatement&      stmt   = mkExprStatement(call);

        record(instrument::Extension(node, stmt));
      }

      void replaceKernelCallIfNeeded(SgCudaKernelCallExp& n)
      {
        FunctionReplacementMap::iterator pos = replmap.find(XPLPragma::KERNEL_LAUNCH);

        if (pos == replmap.end()) return;

        // replace cuda kernel launches
        SgExprListExp&          args = mkArgList();
        CloningBackInserter     inserter(args);

        // copy kernel configuration
        SgCudaKernelExecConfig& kernelConfig = SG_DEREF(n.get_exec_config());

        inserter.push_back(kernelConfig.get_grid());
        inserter.push_back(kernelConfig.get_blocks());
        inserter.push_back(kernelConfig.get_shared(), 0);
        inserter.push_back(kernelConfig.get_stream(), 0);

        // pass kernel function
        inserter.push_back(n.get_function());

        // copy parameters
        SgExpressionPtrList&    origArgs = su::arglist(n);

        std::copy(origArgs.begin(), origArgs.end(), inserter);

        // assemble call to launch wrapper
        SgFunctionRefExp&       tgt  = mkFunctionTarget(SG_DEREF(pos->second));
        SgCallExpression&       call = mkFunctionCallExp(tgt, args);

        // record instrumentation
        record( instrument::repl(n, call) );
      }


      void handle(SgCudaKernelCallExp& n)
      {
        handle(static_cast<SgFunctionCallExp&>(n));

        synchronizeKernelIfNeeded(n);
        replaceKernelCallIfNeeded(n);
      }


      void executeTransformation()
      {
        size_t i = 0;

        for (instrument::AnyTransform& t : replacements)
        {
          t.execute();

          if ((i % 32) == 0)
            display() << '\r' << i << "   ";

          ++i;
        }
      }

      size_t numTransforms() const { return replacements.size(); }

    private:
      void descend(SgNode& n);

      template <class MemFn, class ...Args>
      bool defaults(MemFn memfn, Args... args)
      {
        (this->*memfn)(std::forward<Args>(args)...);
        return true;
      }

      SgFunctionDeclaration& rd()  { return SG_DEREF(trRead); }
      SgFunctionDeclaration& wr()  { return SG_DEREF(trWrite); }
      SgFunctionDeclaration& rw()  { return SG_DEREF(trReadWrite); }

      /// returns the appropriate wrapper function for the expression
      SgFunctionDeclaration& wrapper(SgExpression&);

    
      // data members

      bool                   cudaKernelsRequireSynchronize = false;
      //~ bool                   enabled                       = false;
      bool                   xplactive                     = false;

      SgFunctionDeclaration* trRead                        = nullptr;
      SgFunctionDeclaration* trWrite                       = nullptr;
      SgFunctionDeclaration* trReadWrite                   = nullptr;

      FunctionReplacementMap replmap;
      InstrumentationStore   replacements;
      ScopeList              excludedScopes;

      // a simple tag
      // - false is required to start matching patterns.
      static constexpr bool matchFirstOf = /* do not change */ false;
  };

  void Explorer::descend(SgNode& n)
  {
    *this = std::move(sg::traverseChildren(std::move(*this), n));
  }

  void Explorer::handleExpr(SgExpression& n)
  {
    descend(n);

    instrumentExpr(n);
  }

  InstrumentationGuide::what receiverGuide(SgExpression& n);

  struct MemberAccessGuide : sg::DispatchHandler<InstrumentationGuide::what>
  {
    typedef sg::DispatchHandler<InstrumentationGuide::what> base;

    MemberAccessGuide()
    : base(InstrumentationGuide::read)
    {}

    void handle(SgNode& n)     { SG_UNEXPECTED_NODE(n); }
    void handle(SgExpression&) { /* default handling */ }
    void handle(SgDotExp& n)   { res = receiverGuide(instrument::lhs(n)); }
    void handle(SgVarRefExp&)  { res = InstrumentationGuide::skip; }
  };

  InstrumentationGuide::what receiverGuide(SgExpression& n)
  {
    return sg::dispatch(MemberAccessGuide(), &n);
  }

  void Explorer::handleMemberAccess(SgBinaryOp& n)
  {
    SgExpression& receiver = instrument::lhs(n);
    SgExpression& member   = instrument::rhs(n);

    // do not instrument the callee expression
    //   e.g., x.f() .. x.f remains uninstrumented
    if (isSgMemberFunctionRefExp(&member))
      return;

    if (isCudaCoordinate(receiver))
      return;

    annotate(receiver, receiverGuide(receiver));
    handleExpr(n);
  }

  InstrumentationGuide::what 
  arrayAccessGuide(SgExpression& n)
  {
    if (isSgPntrArrRefExp(&n)) return InstrumentationGuide::skip;

    return InstrumentationGuide::read;
  }

  void Explorer::handle(SgPntrArrRefExp& n)
  {
    SgExpression& lhs = instrument::lhs(n);
    SgExpression& rhs = instrument::rhs(n);

    annotate(lhs, arrayAccessGuide(lhs));
    annotate(rhs, InstrumentationGuide::read);
    handleExpr(n);
  }

  void
  Explorer::functionReplacement(XPLPragma::FunctionReplacement repl)
  {
    replmap[repl.first] = repl.second;
  }

  // we do not want to instrument
  //   lambda captures and
  struct InstrumentableContext : sg::DispatchHandler<bool>
  {
    typedef sg::DispatchHandler<bool> base;

    InstrumentableContext()
    : base(true)
    {}

    void handle(SgNode& n)        { SG_UNEXPECTED_NODE(n); }
    void handle(SgExpression&)    { /* base case */ }
    void handle(SgStatement&)     { /* base case */ }
    void handle(SgLambdaCapture&) { res = false; }
    void handle(SgExprStatement&) { res = false; }
  };

  bool instrumentableContext(SgExpression& n)
  {
    return (  note(n) != InstrumentationGuide::skip
           && sg::dispatch(InstrumentableContext(), n.get_parent())
           );
  }

  struct ScopeStringAssembler : sg::DispatchHandler<std::string>
  {
    void handle(SgNode& n)           { SG_UNEXPECTED_NODE(n); }
    
    // unnamed scopes (should be subsumed by SgScopeStatement)
    //~ void handle(SgBasicBlock& n)     { res = eval(&n); }
    //~ void handle(SgIfStmt& n)         { res = eval(&n); }
    //~ void handle(SgDoWhileStmt& n)    { res = eval(&n); }
    void handle(SgScopeStatement& n) 
    { 
      res = evalScope(&n); 
    }
    
    void handle(SgGlobal&)           { /* return empty string */ }
        
    void handle(SgClassDeclaration& n) 
    { 
      res = evalScope(&n) + SCOPEQUAL + n.get_name(); 
    }
    
    void handle(SgClassDefinition& n) 
    { 
      res = eval(n.get_declaration());
    }
    
    void handle(SgNamespaceDeclarationStatement& n) 
    { 
      res = evalScope(&n) + SCOPEQUAL + n.get_name(); 
    }
    
    void handle(SgNamespaceDefinitionStatement& n) 
    { 
      res = eval(n.get_namespaceDeclaration());
    }
    
    void handle(SgFunctionDeclaration& n)
    {
      if (&n != n.get_firstNondefiningDeclaration())
      {
        res = eval(n.get_firstNondefiningDeclaration());
        return;
      }

      res = evalScope(&n) + SCOPEQUAL + n.get_name();
    }
    
    void handle(SgTemplateInstantiationFunctionDecl& n)
    {
      res = eval(n.get_templateDeclaration());
    }
    
    void handle(SgTemplateInstantiationMemberFunctionDecl& n)
    {
      res = eval(n.get_templateDeclaration());
    } 
    
    void handle(SgFunctionDefinition& n) 
    { 
      res = eval(n.get_declaration());
    } 
    
    static
    std::string eval(SgNode* n);
    
    static
    std::string evalScope(SgNode* n);
    
    const static std::string SCOPEQUAL;  
  };
  
  const std::string ScopeStringAssembler::SCOPEQUAL{"::"};  
  
  std::string ScopeStringAssembler::eval(SgNode* n)
  {
    return sg::dispatch(ScopeStringAssembler(), n);
  }
  
  std::string ScopeStringAssembler::evalScope(SgNode* n)
  {
    return eval(si::getEnclosingScope(n));
  }
  
  std::string scopeString(SgExpression& n)
  {
    return ScopeStringAssembler::evalScope(&n);
  }

  bool Explorer::excludedScope(SgExpression& n) const
  {
    ScopeList::const_iterator zz       = excludedScopes.end();
    const std::string         scoperep = scopeString(n);
    
    return zz != std::find_if( excludedScopes.begin(), zz, 
                               [&scoperep](const std::string& excluded) -> bool
                               {
                                 //~ logInfo() << excluded << " <excl?> " << scoperep << std::endl;
                                 
                                 return scoperep.find(excluded) != std::string::npos;
                               }
                             );
  }

  SgFunctionDeclaration&
  Explorer::wrapper(SgExpression& n)
  {
    if (  isSgExprListExp(n.get_parent())
       || note(n) == InstrumentationGuide::read
       || !n.isUsedAsLValue()
       )
      return rd();

    if (note(n) == InstrumentationGuide::readWrite)
      return rw();

    return wr();
  }

  void Explorer::instrumentExpr(SgExpression& n)
  {
    // skip instrumentation if not active
    if (!xplactive) return;

    // instrument read operations:
    //   => all non-temporary values (l-values)
    //      that are not used in a write context.
    if (!n.isLValue()) return;

    // skip over contexts that are not relevant.
    if (!instrumentableContext(n)) return;
    
    // skip if scope has been excluded
    if (excludedScope(n)) return;
    
    record( instrument::wrap(n, SG_DEREF(sb::buildFunctionRefExp(&wrapper(n)))) );
  }



  struct XPL : Rose::PluginAction
  {
    // This is mandatory: providing work in your plugin
    // Do actual work after ParseArgs();
    void process (SgProject* n) ROSE_OVERRIDE
    {
      display() << "\nXPL: analyzing..." << std::endl;
      Explorer expl = sg::traverseChildren(Explorer(), sg::deref(n));
      display() << '\r' << expl.numTransforms() << " transformations found." << std::endl;

      display() << "\nXPL: transforming..." << std::endl;
      expl.executeTransformation();
      display() << '\r' << expl.numTransforms() << " transformations executed." << std::endl;

      display() << "\nXPL: done." << std::endl;
    } // end process()  
  };
}

//Step 2: Declare a plugin entry with a unique name
//        Register it under a unique action name plus some description
static Rose::PluginRegistry::Add<tracer::XPL> xpltracerName("xpltracer", "instruments CUDA code");


#if OBSOLETE_CODE
  struct ArrayAccessible : sg::DispatchHandler<bool>
  {
    void handle(SgNode& n)      { SG_UNEXPECTED_NODE(n); }

    void handle(SgType&)        { /* default result = false */ }

    void handle(SgArrayType&)   { res = true; }
    void handle(SgPointerType&) { res = true; }
  };

  bool isArrayAccessible(SgType& ty)
  {
    return sg::dispatch(ArrayAccessible(), &ty);
  }

  bool isArrayType(SgType& ty)
  {
    SgType& under = su::skipTypes(ty, su::typedefs | su::modifiers | su::references );

    return isArrayAccessible(under);
  }

  bool hasArrayType(SgExpression& n)
  {
    return isArrayType(su::type(n));
  }

  struct IsLocal : sg::DispatchHandler<bool>
  {
    typedef sg::DispatchHandler<bool> base;

    IsLocal()
    : base(true)
    {}

    void handle(const SgNode& n)                       { SG_UNEXPECTED_NODE(n); }
    void handle(const SgScopeStatement&)               { /* base case */ }
    void handle(const SgGlobal&)                       { res = false; }
    void handle(const SgNamespaceDefinitionStatement&) { res = false; }
  };

  bool isLocal(SgVarRefExp& n)
  {
    SgInitializedName& var = su::keyDecl(n);

    return sg::dispatch(IsLocal(), var.get_scope());
  }

  struct NeedsInstrumentation : sg::DispatchHandler<bool>
  {
      void handle(const SgNode& n)          { SG_UNEXPECTED_NODE(n); }
      void handle(const SgExpression&)      { /* default */ }
      void handle(const SgPointerDerefExp&) { res = true;   }
      void handle(const SgPntrArrRefExp&)   { res = true;   }
  };

  bool needsInstrumentation(const SgNode& n)
  {
    return sg::dispatch(NeedsInstrumentation(), &n);
  }
#endif /* OBSOLETE_CODE */
