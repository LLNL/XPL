

#ifndef _XPL_PRAGMAS_H
#define _XPL_PRAGMAS_H 1

#include "rose.h"
#include "plugin.h"

// optional headers
#include "sageGeneric.h"
#include "sageBuilder.h"
#include "sageUtility.h"
#include "sageInterface.h"

#include "transformations.h"

namespace tracer
{
  /// \brief   A class to handle xpl tracer specific pragmas (XPLPragmas)
  /// \details pragma syntax is:
  ///          #pragma xpl diagnostic func ['(' verbatim-args.. ; var1 {',' varN } '}]
  ///          - verbatim-args.. arguments that are copied verbatim to func
  ///          - varX are valid C++ expressions. The print
  ///            function will use these expressions to identify the logged data.
  ///          #pragma xpl kernel synchronize
  ///          #pragma xpl replace funcName
  struct XPLPragma
  {
      typedef std::pair<std::string, SgFunctionDeclaration*> FunctionReplacement;

      // \todo generalize "diagnostic" and "sync_kernel" into
      //       "before", "after", "call-here" instrumentations of a call.
      enum PragmaKind 
      { 
        undefined, 
        xplon, 
        xploff, 
        diagnostic, 
        replace, 
        sync_kernel, 
        ignore_diagnostic, 
        exclude_scopes,
        unknown, 
        noxpl 
      };

      /// tests whether the string @ref s constitutes an XPL pragma.
      ///   consumes the pragma descriptor (e.g., xpl replace),
      ///   and sets the kind.
      void parsePrefix();

      XPLPragma(SgPragmaDeclaration& n, SgPragma& info)
      : node(n), kind(undefined), text(info.get_pragma())
      {
        parsePrefix();
      }

      /// returns if this is a recognized XPL pragma
      //operator bool() const { return kind > undefined && kind < unknown; }

      /// returns the kind of pragma
      PragmaKind which() const { return kind; }

      //
      // pragma specific functions

      /// Parses the pragma text and computes a transformation to embed
      ///   the generated code (i.e, diagnostic)
      /// \returns a transformation action
      instrument::Extension action();
      
      /// sets the new diagnostic function ignore list
      void updateIgnoreList();
      
      /// updates the excluded scope lists @ref excluded
      void updateExcludedScopes(std::vector<std::string>& excluded);

      /// Returns a description of functional replacement (i.e., replace)
      FunctionReplacement functionReplacement();

      static const std::string KERNEL_LAUNCH;

    private:
      SgStatement& node;
      PragmaKind   kind;
      std::string  text;

      XPLPragma()                            = delete;
      XPLPragma(const XPLPragma&)            = delete;
      XPLPragma(XPLPragma&&)                 = delete;
      XPLPragma& operator=(XPLPragma&&)      = delete;
      XPLPragma& operator=(const XPLPragma&) = delete;
  };
}

#endif /* _XPL_PRAGMAS_H */
