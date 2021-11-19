
#ifndef _TRANSFORMATIONS_H
#define _TRANSFORMATIONS_H 1

#include <memory>

#include "rose.h"
#include "plugin.h"


// optional headers
#include "sageGeneric.h"
#include "sageBuilder.h"
#include "sageUtility.h"
#include "sageInterface.h"

namespace
{
  static constexpr bool MARK_TRANSFORMED = true;
  static constexpr bool PRINT_HEADERS    = false;
  
  inline
  SgExpression&
  mkFunctionTarget(const std::string& funname, SgScopeStatement& scope)
  {
    return SG_DEREF(SageBuilder::buildOpaqueVarRefExp(funname, &scope));
  }

  inline
  SgFunctionRefExp&
  mkFunctionTarget(SgFunctionDeclaration& decl)
  {
    return SG_DEREF(SageBuilder::buildFunctionRefExp(&decl));
  }

  inline
  SgExprListExp&
  mkArgList()
  {
    return SG_DEREF(SageBuilder::buildExprListExp());
  }

  inline
  SgCallExpression&
  mkFunctionCallExp(SgExpression& tgt, SgExprListExp& args)
  {
    return SG_DEREF(SageBuilder::buildFunctionCallExp(&tgt, &args));
  }

  inline
  SgExprStatement&
  mkExprStatement(SgExpression& expr)
  {
    return SG_DEREF(SageBuilder::buildExprStatement(&expr));
  }
}


namespace tracer
{
  namespace instrument
  {
    //
    // accessor functions
    namespace
    {
      inline
      SgExpression& lhs(SgBinaryOp& n)                  { return SG_DEREF(n.get_lhs_operand()); }

      inline
      SgExpression& rhs(SgBinaryOp& n)                  { return SG_DEREF(n.get_rhs_operand()); }

      inline
      SgExpression& operand(SgUnaryOp& n)               { return SG_DEREF(n.get_operand()); }

      inline
      SgExpression& initializer(SgAssignInitializer& n) { return SG_DEREF(n.get_operand()); }

      inline
      SgExpression& identity(SgExpression& n)           { return n; }
    }

    //
    // Generic Polymorphic actions

    /// Abstract base class to define essential polymorphic Transform functions.
      struct BaseTransform
      {
        virtual ~BaseTransform() = default;

        virtual void execute() const = 0;
        virtual SgLocatedNode& modifiedNode() const = 0;
        virtual BaseTransform* clone() const = 0;
      };

      /// Polymorphic wrapper for concrete actions.
      template <class ConcreteTransform>
      struct PolyTransform : BaseTransform
      {
        explicit
        PolyTransform(const ConcreteTransform& concreteTf)
        : tf(concreteTf)
        {}

        void execute() const ROSE_OVERRIDE
        {
          tf.execute();
        }
        
        SgLocatedNode& modifiedNode() const ROSE_OVERRIDE
        {
          return tf.modifiedNode();
        }        

        PolyTransform<ConcreteTransform>*
        clone() const ROSE_OVERRIDE
        {
          return new PolyTransform(*this);
        }

        ConcreteTransform tf;
      };
      
      inline
      SgDeclarationStatement& scopeLevelDeclaration(SgLocatedNode& n)
      {
        SgDeclarationStatement& res  = sg::ancestor<SgDeclarationStatement>(n);
        const bool              skip = isSgCtorInitializerList(&res);
        
        return skip ? scopeLevelDeclaration(res) : res;
      }

      /// Generic walue wrapper around polymorphic actions.
      struct AnyTransform
      {
        // wrapping value ctors
        template <class ConcreteTransform>
        AnyTransform(const ConcreteTransform& a)
        : tf(new PolyTransform<ConcreteTransform>(a))
        {}

        template <class ConcreteTransform>
        AnyTransform(ConcreteTransform&& a)
        : tf(new PolyTransform<ConcreteTransform>(std::move(a)))
        {}

        // move ctor + assignment
        AnyTransform(AnyTransform&& other)                 = default;
        AnyTransform& operator=(AnyTransform&& other)      = default;

        // copy ctor + assignment
        AnyTransform(const AnyTransform& other)            = delete;
        AnyTransform& operator=(const AnyTransform& other) = delete;

        // dtor
        ~AnyTransform()                                    = default;
        
        static 
        bool inHeaderFile(const Sg_File_Info& info)
        {
          std::string filename = info.get_raw_filename();
          
          return (  (filename.find(".h") != std::string::npos)
                 || (filename.find(".H") != std::string::npos)
                 || (filename.find(".")  == std::string::npos) 
                 );
        }
        
        // business logic
        void markTransformed(SgDeclarationStatement& dcl) const
        {
          dcl.setTransformation();
          
          SageInterface::setSourcePositionForTransformation(&dcl);
        }

        void markTransformed() const
        {
          if (!MARK_TRANSFORMED) return;
          
          SgLocatedNode&          modnd  = tf->modifiedNode();
          SgDeclarationStatement& defdcl = scopeLevelDeclaration(modnd);
          Sg_File_Info&           info   = SG_DEREF(defdcl.get_file_info());
          
          // if the transformation is from a header file, skip it unless PRINT_HEADERS is set
          if ((!PRINT_HEADERS) && inHeaderFile(info))
            return;
                    
          markTransformed(defdcl);

          if (defdcl.isCompilerGenerated())
          { 
            SgDeclarationStatement& nondef = SG_DEREF(defdcl.get_firstNondefiningDeclaration());
            
            markTransformed(nondef);
          }          
        }

        void execute() const 
        { 
          tf->execute(); 
          
          markTransformed();
        }      

        std::unique_ptr<BaseTransform> tf;
      };

      template <class U, class V>
      bool equals(U, V)
      {
        return false;
      }

      template <class U>
      bool equals(U lhs, U rhs)
      {
        return lhs == rhs;
      }

    template <class SageNode, class Accessor>
    struct Wrapper
    {
      explicit
      Wrapper(SageNode& n, Accessor accfn, SgExpression& wrapfn)
      : astnode(n), accessor(accfn), wrapperfn(wrapfn), modnode(nullptr)
      {}

      void execute() const
      {
        namespace sb = SageBuilder;
        namespace si = SageInterface;

        SgExpression*  current = nullptr;
        SgExpression*  newnode = nullptr;
        SgExpression&  oldexp  = accessor(astnode);

        // fast mode
        if (&oldexp == &astnode)
        {
          SgExpression& dummy = SG_DEREF(sb::buildNullExpression());

          si::replaceExpression(&astnode, &dummy, true /* keep */);
          current = &dummy;
          newnode = &oldexp;
        }
        else
        {
          // uses deep copy
          current = &oldexp;
          newnode = si::deepCopy(&oldexp);
        }

        ROSE_ASSERT(current && newnode);
        SgExprListExp& args   = SG_DEREF(sb::buildExprListExp(newnode));
        SgExpression&  wrpexp = SG_DEREF(sb::buildFunctionCallExp(&wrapperfn, &args));

        si::replaceExpression( current, &wrpexp, false /* do not keep */ );
       
        if (MARK_TRANSFORMED) modnode = &wrpexp;       
      }
      
      SgLocatedNode& modifiedNode() const
      {
        return SG_DEREF(modnode);
      }

      SageNode&             astnode;
      Accessor              accessor;
      SgExpression&         wrapperfn;
      SgExpression mutable* modnode;
    };

    template <class SageNode, class Accessor>
    static inline
    Wrapper<SageNode, Accessor>
    wrap(SageNode& n, Accessor accfn, SgExpression& wrapfn)
    {
      return Wrapper<SageNode, Accessor>(n, accfn, wrapfn);
    }

    template <class SageNode>
    static inline
    auto
    wrap(SageNode& n, SgExpression& wrapfn) -> decltype( wrap(n, identity, wrapfn) )
    {
      return wrap(n, identity, wrapfn);
    }

    struct Replacer
    {
      explicit
      Replacer(SgExpression& oldnode, SgExpression& newnode)
      : oldexp(oldnode), rplexp(newnode)
      {}

      void execute() const
      {
        SageInterface::replaceExpression( &oldexp, &rplexp );
      }
      
      SgLocatedNode& modifiedNode() const
      {
        return rplexp;
      }

      SgExpression& oldexp;
      SgExpression& rplexp;
    };

    static inline
    Replacer
    repl(SgExpression& oldnode, SgExpression& newnode)
    {
      return Replacer(oldnode, newnode);
    }

    /// inserts a new statement
    struct Extension
    {
        Extension()
        : existing(nullptr), addition(nullptr)
        {}
         
        Extension(SgStatement& pred, SgStatement& stmt)
        : existing(&pred), addition(&stmt)
        {}
        
        bool valid()   const { return existing && addition; }
  
        void execute() const
        {
          ROSE_ASSERT(valid());
          
          SageInterface::insertStatement(existing, addition, false /* insert after existing */);
        }
        
        SgLocatedNode& modifiedNode() const
        {
          return SG_DEREF(addition);
        }
      
      private:
        SgStatement* existing;
        SgStatement* addition;
    };
  }  // namespace instrument
}  // namespace tracer

#endif /* _TRANSFORMATIONS_H */
