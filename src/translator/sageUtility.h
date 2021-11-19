#ifndef UTILS_H
#define UTILS_H

#include <string>

#include <boost/lexical_cast.hpp>

#include "rose.h"

/// copied from Matlab project
namespace SageUtil
{
#if 0
  /*
    Build Empty parameters
   */
  SgExprListExp* buildEmptyParams();

  /*
Returns a string of types separated by commas from a list of types
*/
  std::string getFunctionCallTypeString(Rose_STL_Container<SgType*> typeList);

  /*
    Replace all variables in the given SgStatement that have the same name as given variable by the new expression
  */
  void replaceVariable(SgStatement *statement, SgVarRefExp *variable, SgExpression *expression);

/*
 * Get the rows of a SgMatrixExp as a vector of ExprListExp
*/
  Rose_STL_Container<SgExprListExp*> getMatrixRows(SgMatrixExp *matrix);


  /*
  *Create a template variable declaration with your provided types
  *Eg. You may want to create Matrix<int> x;
  *And you may not have an existing symbol for Matrix<int>
  */
  SgVariableDeclaration *createOpaqueTemplateObject(std::string varName, std::string className, std::string type, SgScopeStatement *scope);

  /*
   *Create a member function call
   *This function looks for the function symbol in the given className
   *The function should exist in the class
   *The class should be #included or present in the source file parsed by frontend
   */
  SgFunctionCallExp *createMemberFunctionCall(std::string className, SgExpression *objectExpression, std::string functionName, SgExprListExp *params, SgScopeStatement *scope);


  /*
   * Returns start:stride:end as a SgExprListExp
   */
  SgExprListExp *getExprListExpFromRangeExp(SgRangeExp *rangeExp);

  SgFunctionCallExp *createFunctionCall(std::string functionName, SgScopeStatement *scope, SgExprListExp *parameters);


  /// returns name of symbol
  SgName nameOf(const SgSymbol& varsy);

  /// returns name of symbol
  SgName nameOf(const SgVarRefExp& var_ref);

  /// returns name of symbol
  SgName nameOf(const SgVarRefExp* var_ref);
#endif

namespace
{
  template <class T>
  struct ScopedUpdate
  {
      ScopedUpdate(T& mem, const T& newval)
      : adr(mem), val(mem)
      {
        adr = newval;
      }

      ~ScopedUpdate()
      {
        adr = val;
      }

    private:
      T& adr;
      T  val;

      ScopedUpdate()                               = delete;
      ScopedUpdate(const ScopedUpdate&)            = delete;
      ScopedUpdate(ScopedUpdate&&)                 = delete;
      ScopedUpdate& operator=(const ScopedUpdate&) = delete;
      ScopedUpdate& operator=(ScopedUpdate&&)      = delete;
  };  
  
  inline
  std::string str(const SgNode* n)
  {
    return (n? n->unparseToString() : std::string("<null>"));
  }

  inline
  std::string str(const Sg_File_Info* n)
  {
    if (!n) return "?";

    return ( n->get_filenameString()
           + " : "
           + boost::lexical_cast<std::string>(n->get_line())
           );
  }

  template <class ErrorClass = std::runtime_error>
  inline
  void chk(bool success, const std::string& a, std::string b = "", std::string c = "")
  {
    if (!success) throw ErrorClass(a + b + c);
  }

  template <class ErrorClass = std::runtime_error>
  inline
  void chk(bool success, const std::string& a, SgNode& n)
  {
    if (!success) throw ErrorClass(a + n.unparseToString());
  }


  /// returns the parameter list of a function call
  inline
  const SgInitializedNamePtrList& paramlist(const SgFunctionDeclaration& fndcl)
  {
    return fndcl.get_args();
  }

  /// returns the parameter list of a function call
  inline
  const SgInitializedNamePtrList& paramlist(const SgFunctionDefinition& fndef)
  {
    return paramlist(SG_DEREF(fndef.get_declaration()));
  }

  /// returns the argument list of a function call
  inline
  SgExpressionPtrList& arglist(const SgCallExpression& call)
  {
    SgExprListExp& args = sg::deref(call.get_args());

    return args.get_expressions();
  }

  /// returns the argument list of a function call
  inline
  SgExpressionPtrList& arglist(const SgCallExpression* call)
  {
    return arglist(sg::deref(call));
  }

  /// returns the n-th argument of a function call
  inline
  SgExpression& argN(const SgCallExpression& call, size_t n)
  {
    return sg::deref(arglist(call).at(n));
  }

  /// returns the n-th argument of a function call
  inline
  SgExpression& argN(const SgCallExpression* call, size_t n)
  {
    return argN(sg::deref(call), n);
  }

  /// returns the declaration associated with a variable reference
  inline
  SgInitializedName& decl(const SgVarRefExp& n)
  {
    SgVariableSymbol& sym = SG_DEREF(n.get_symbol());

    return SG_DEREF(sym.get_declaration());
  }

  /// returns the declaration associated with a function reference
  /// \note assers that the declaration is available
  inline
  SgFunctionDeclaration& decl(const SgFunctionRefExp& n)
  {
    return SG_DEREF(n.getAssociatedFunctionDeclaration());
  }


  /// Returns a translation unit's key declaration of a given declaration.
  /// \tparam a type that is a SgDeclarationStatement or inherited from it.
  /// \param  dcl a valid sage declaration (&dcl must not be nullptr).
  /// \details
  ///    a key declaration is the defining declaration (if available),
  ///    or the first non-defining declaration (if a definition is not available).
  template <class SageDecl>
  inline
  SageDecl& keyDecl(const SageDecl& dcl)
  {
    SgDeclarationStatement* keydcl = dcl.get_definingDeclaration();
    if (keydcl) return SG_ASSERT_TYPE(SageDecl, *keydcl);

    keydcl = dcl.get_firstNondefiningDeclaration();
    return SG_ASSERT_TYPE(SageDecl, SG_DEREF(keydcl));
  }

  /// returns the key declaration associated with a function reference
  /// \note asserts that the function is available
  inline
  SgFunctionDeclaration& keyDecl(const SgFunctionRefExp& n)
  {
    return keyDecl(decl(n));
  }

  /// returns the initialized name of a variable declaration
  /// \note asserts that a variable declaration only initializes
  ///       a single variable.
  inline
  SgInitializedName& initName(const SgVariableDeclaration& n)
  {
    const SgInitializedNamePtrList& varlst = n.get_variables();
    ROSE_ASSERT(varlst.size() == 1);

    return SG_DEREF(varlst.at(0));
  }

  /// returns the key declaration of a variable declaration
  /// \note the key declaration of a variable is the initialized
  ///       name (SgInitializedName) of the defining declaration.
  ///       If no defining declaration is available, the initialized
  ///       name of the first non-defining declaration is used.
  inline
  SgInitializedName& keyDecl(const SgVariableDeclaration& n)
  {
    SgDeclarationStatement* defdcl   = n.get_definingDeclaration();

    if (defdcl)
      return initName(SG_ASSERT_TYPE(SgVariableDeclaration, *defdcl));

    defdcl = n.get_firstNondefiningDeclaration();
    return initName(SG_ASSERT_TYPE(SgVariableDeclaration, SG_DEREF(defdcl)));
  }

  /// auxiliary struct for finding key variable declarations
  struct KeyDeclFinder : sg::DispatchHandler<SgInitializedName*>
  {
      explicit
      KeyDeclFinder(const SgInitializedName& dcl)
      : var(dcl)
      {}

      void handle(SgNode& n)                { SG_UNEXPECTED_NODE(n); }
      void handle(SgVariableDeclaration& n) { res = &keyDecl(n); }
      void handle(SgFunctionParameterList&)
      {
        // \todo should this be the parameter in a key-declaration
        //       or the declaration that defines the default argument.
        res = &const_cast<SgInitializedName&>(var);
      }

    private:
      const SgInitializedName& var;
  };

  /// returns the key declaration of an initializd name
  inline
  SgInitializedName& keyDecl(const SgInitializedName& n)
  {
    SgInitializedName* res = sg::dispatch(KeyDeclFinder(n), n.get_declaration());

    return SG_DEREF(res);
  }

  /// returns the key declaration of a variable reference
  inline
  SgInitializedName& keyDecl(const SgVarRefExp& n)
  {
    return keyDecl(decl(n));
  }

  /// gives key declarations a distinct type
  ///   (e.g., for use inside a map, or to not have to require the same info...)
  template <class SageDecl>
  struct KeyDecl
  {
      // implicit
      KeyDecl(SageDecl& n)
      : keydcl(keyDecl(n))
      {}

      KeyDecl(const KeyDecl&) = default;
      KeyDecl(KeyDecl&&)      = default;

      SageDecl& decl()     const   { return keydcl; }

      SageDecl* operator->() const { return &keydcl; }
    private:
      SageDecl& keydcl;

      KeyDecl()                          = delete;
      KeyDecl& operator=(KeyDecl&&)      = delete;
      KeyDecl& operator=(const KeyDecl&) = delete;
  };

  enum TypeSkipMode : size_t
  {
    modifiers  = 1 << 0,
    lvaluerefs = 1 << 1,
    rvaluerefs = 1 << 2,
    references = (lvaluerefs + rvaluerefs),
    typedefs   = 1 << 3,
    usingdecl  = 1 << 4,
    aliases    = (typedefs + usingdecl),
    pointers   = 1 << 5,
    arrays     = 1 << 6,
  };

  struct TypeSkipper : sg::DispatchHandler<SgType*>
  {
    typedef sg::DispatchHandler<SgType*> base;

    explicit
    TypeSkipper(TypeSkipMode m)
    : base(), mode(m)
    {}

    SgType* recurse(SgNode* ty) { return sg::dispatch(TypeSkipper(mode), ty); }

    void handle(SgNode& n)      
    { 
      res = isSgType(&n);
      
      if (!res) SG_UNEXPECTED_NODE(n); 
    }

    void handle(SgType& n)      { res = &n; }

    void handle(SgModifierType& n)
    {
      res = (mode & modifiers) ? recurse(n.get_base_type()) : &n;
    }

    void handle(SgReferenceType& n)
    {
      res = (mode & lvaluerefs) ? recurse(n.get_base_type()) : &n;
    }

    void handle(SgRvalueReferenceType& n)
    {
      res = (mode & rvaluerefs) ? recurse(n.get_base_type()) : &n;
    }

    void handle(SgPointerType& n)
    {
      res = (mode & pointers) ? recurse(n.get_base_type()) : &n;
    }

    void handle(SgArrayType& n)
    {
      res = (mode & arrays) ? recurse(n.get_base_type()) : &n;
    }

    void handle(SgTypedefType& n)
    {
      res = (mode & typedefs) ? recurse(n.get_base_type()) : &n;
    }

    TypeSkipMode mode;
  };

  inline
  SgType& skipTypes(SgType& ty, size_t mode)
  {
    SgType* res = sg::dispatch(TypeSkipper(TypeSkipMode(mode)), &ty);
    return SG_DEREF(res);
  }

  /// returns a type without modifiers
  inline
  SgType& skipTypeModifier(SgType& t)
  {
    return skipTypes(t, modifiers);
  }

  inline
  SgType& type(SgExpression& n)
  {
    return SG_DEREF(n.get_type());
  }
  
  /// returns name of symbol
  inline
  SgName nameOf(const SgSymbol& varsy)
  {
    return varsy.get_name();
  }

  inline
  SgName nameOf(const SgVarRefExp& var_ref)
  {
    return nameOf(SG_DEREF(var_ref.get_symbol()));
  }
} // anonymous namespace
} // namespace SageUtil
#endif
