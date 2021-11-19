#include <string>
#include <set>

#include "xpl-pragmas.h"

//~ #include <boost/algorithm/string.hpp>
//~ #include <boost/tokenizer.hpp>

namespace su = SageUtil;
namespace sb = SageBuilder;
namespace si = SageInterface;

namespace tracer
{
  namespace
  {
    std::set<std::string> diagnosticFunctionIgnoreList;
    
    inline
    std::string ltrim(const std::string& s)
    {
      const size_t trimpos = s.find_first_not_of(' ');

      if (trimpos == std::string::npos) return "";

      return s.substr(trimpos);
    }
    
    inline
    std::string rtrim(const std::string& s)
    {
      const size_t trimpos = s.find_last_not_of(' ');

      if (trimpos == std::string::npos) return "";

      //~ std::cerr << "TRM:" << s.substr(0, trimpos+1) << std::endl;
      return s.substr(0, trimpos+1);
    }

    size_t 
    includeArguments(std::string& txt, size_t openpos)
    {
      size_t level = 1;

      do
      {
        const size_t pos = txt.find_first_of("()", openpos+1);

        if (pos == std::string::npos)
          throw std::runtime_error("incorrectly terminated XPL pragma: " + txt);

        if (txt.at(pos) == '(') ++level;
        if (txt.at(pos) == ')') --level;

        openpos = pos;
      } while (level);

/*
      openpos = txt.find_first_of(",;)");
      if (openpos == std::string::npos)
        throw std::runtime_error("incorrectly terminated XPL pragma: " + txt);
*/
      return openpos+1;
    }

    void
    verbatimArguments(std::string& pragma, SgExprListExp& args, SgScopeStatement& scope)
    {
      char delim = '(';

      while (delim != ';' && delim != ')')
      {
        size_t delimpos   = pragma.find_first_of("(,;)");

        if (delimpos == std::string::npos)
          throw std::runtime_error("incorrectly terminated XPL pragma" + pragma);

        if (pragma.at(delimpos) == '(')
          delimpos = includeArguments(pragma, delimpos);

        std::string token = pragma.substr(0, delimpos);

        if (token.size() > 0)
        {
          args.append_expression( sb::buildOpaqueVarRefExp(token, &scope) );
        }

        delim  = pragma.at(delimpos);
        pragma = pragma.substr(delimpos+1);
      }

      pragma = ltrim(pragma);
    }

    void
    processExpression(std::string expr, SgExprListExp& args, SgScopeStatement& scope)
    {
      // \todo create a clean implementation
      std::string argtext = "XPLAllocData(" + expr + ", \"" + expr + "\")";

      args.append_expression( sb::buildOpaqueVarRefExp(argtext, &scope) );
    }

    void
    processVariable(SgType& ty, std::string expr, SgExprListExp& args, SgScopeStatement& scope)
    {
      const size_t        skips    = su::references | su::typedefs | su::modifiers;
      SgType*             basety   = &su::skipTypes(ty, skips);
      SgPointerType*      ptrty    = isSgPointerType(basety);
      std::string         selector = ".";

      // if expr is a pointer variable or field, add it to the list of interesting
      //   allocations
      if (ptrty)
      {
        processExpression(expr, args, scope);
        selector = "->";

        basety = &su::skipTypes( SG_DEREF(ptrty->get_base_type()), skips );
      }

      SgClassType*        clazzty  = isSgClassType(basety);
      if (!clazzty) return;

      // the underlying type is a class, thus look for all reachable
      //   allocations from it.
      SgClassDeclaration& clazzdcl = SG_ASSERT_TYPE( SgClassDeclaration,
                                                     SG_DEREF(clazzty->get_declaration())
                                                   );

      SgClassDeclaration& keydcl   = su::keyDecl(clazzdcl);
      SgClassDefinition*  clazzdef = keydcl.get_definition();
      if (!clazzdef) return;

      // traverse fields with pointer type recursively
      for (SgDeclarationStatement* n : clazzdef->get_members())
      {
        SgVariableDeclaration*  var = isSgVariableDeclaration(n);
        if (!var) continue;

        for (SgInitializedName* field : var->get_variables())
        {
          SgType&        dclty  = SG_DEREF(field->get_type());
          SgPointerType* ptrfld = isSgPointerType(&su::skipTypes(dclty, skips));

          if (!ptrfld) continue;

          processVariable(*ptrfld, "(" + expr + ")" + selector + field->get_name(), args, scope);
        }
      }

      // \todo handle arrays of pointers
      // \todo go through base class member lists
    }

    void
    processVariableExpr(std::string expr, SgExprListExp& args, SgScopeStatement& scope)
    {
      expr = ltrim(expr);

      if (expr.size() == 0) return;

      SgVariableSymbol* varsy = si::lookupVariableSymbolInParentScopes(expr, &scope);

      if (!varsy)
      {
        processExpression(expr, args, scope);
        return;
      }

      processVariable(SG_DEREF(varsy->get_type()), expr, args, scope);
    }

    void
    userLevelObjects(std::string& pragma, SgExprListExp& args, SgScopeStatement& scope)
    {
      if (pragma.size() == 0) return;

      char delim = ';';

      while (delim != ')')
      {
        const size_t delimpos = pragma.find_first_of(" ,)");

        if (delimpos == std::string::npos)
          throw std::runtime_error("incorrectly terminated XPL pragma");

        std::string  token    = pragma.substr(0, delimpos);

        if (token.size() > 0)
        {
          processVariableExpr(token, args, scope);
        }

        delim  = pragma.at(delimpos);
        pragma = pragma.substr(delimpos+1);
      }
    }

    bool consume(const std::string& candToken, std::string& text)
    {
      const size_t                toklen = candToken.size();

      if (text.size() < toklen) return false;

      std::string::const_iterator eot = candToken.end();

      if (std::mismatch(candToken.begin(), eot, text.begin()).first != eot)
        return false;

      text = ltrim(text.substr(toklen));
      return true;
    }

/*
    void consumeExpected(const std::string& requiredToken, std::string& text)
    {
      const bool consumed = consume(requiredToken, text);

      if (consumed) return;

      throw std::runtime_error( std::string("xpl macro, expected token ")
                              + requiredToken + "; got" + text
                              );
    }
*/
  }


  void
  XPLPragma::parsePrefix()
  {
    static const std::string XPL_PREFIX      = "xpl";
    static const std::string XPL_REPLACE     = "replace";
    static const std::string XPL_IGNORE_DIAG = "diagnostic-ignore";
    static const std::string XPL_DIAGNOSTIC  = "diagnostic";
    static const std::string XPL_SYNC_KERNEL = "kernel-synchronize";
    static const std::string XPL_EXCL_SCOPES = "exclude-scopes";
    static const std::string XPL_INSTR_ON    = "on";
    static const std::string XPL_INSTR_OFF   = "off";

    text = ltrim(text);

    if      (!consume(XPL_PREFIX, text))     kind = noxpl;
    else if (consume(XPL_REPLACE, text))     kind = replace;
    else if (consume(XPL_IGNORE_DIAG, text)) kind = ignore_diagnostic;
    else if (consume(XPL_DIAGNOSTIC, text))  kind = diagnostic;
    else if (consume(XPL_SYNC_KERNEL, text)) kind = sync_kernel;
    else if (consume(XPL_INSTR_ON, text))    kind = xplon;
    else if (consume(XPL_INSTR_OFF, text))   kind = xploff;
    else if (consume(XPL_EXCL_SCOPES, text)) kind = exclude_scopes;
    else                                     kind = unknown;
  }
  
  template <class Inserter>
  void forEachToken(Inserter ins, std::string text, char sep = ',') 
  {
    size_t pos = text.find(sep);
    
    while (pos != std::string::npos)
    {
      ins(rtrim(ltrim(text.substr(0, pos))));
      
      text   = text.substr(pos+1); 
      pos    = text.find(sep);
    }
    
    ins(rtrim(ltrim(text)));
  }
  
  void XPLPragma::updateExcludedScopes(std::vector<std::string>& excluded)
  {
    assert(kind == exclude_scopes);
    
    excluded.clear();
    forEachToken( [&excluded](std::string s) -> void
                  {
                    excluded.push_back(std::move(s));
                  },
                  std::move(text)
                );     
  }

  void
  XPLPragma::updateIgnoreList()
  {
    assert(kind == ignore_diagnostic);
    
    diagnosticFunctionIgnoreList.clear();
    forEachToken( [](std::string s) -> void
                  {
                    diagnosticFunctionIgnoreList.insert(std::move(s));
                  },
                  std::move(text)
                );    
  }

  instrument::Extension
  XPLPragma::action()
  {
    assert(kind == diagnostic);

    const size_t      openparens = text.find_first_of("(");
    if (openparens == std::string::npos)
      throw std::runtime_error("missing parenthesis '(' in XPL diagnostic pragma: " + text);

    const std::string diagnosticfun = rtrim(ltrim(text.substr(0, openparens)));

    if (diagnosticfun.size() == 0)
      throw std::runtime_error("no function name found in XPL diagnostic pragma.");
      
    // ignore some functions
    if (diagnosticFunctionIgnoreList.find(diagnosticfun) != diagnosticFunctionIgnoreList.end())
    {
      //~ std::cerr << "/IGN: " << diagnosticfun << std::endl;
      return instrument::Extension();
    }

    text = text.substr(openparens+1);

    SgScopeStatement& scope  = sg::ancestor<SgScopeStatement>(node);
    SgExpression&     callee = mkFunctionTarget(diagnosticfun, scope);
    SgExprListExp&    args   = mkArgList();

    // process user pragma code
    verbatimArguments(text, args, scope);
    userLevelObjects(text, args, scope);

    SgCallExpression& call    = mkFunctionCallExp(callee, args);
    SgStatement&      stmt    = mkExprStatement(call);

    //~ consumeExpected(")", text);
    if (text.size() != 0)
      throw std::runtime_error("unexpected code after ')' in XPL diagnostic pragma: " + text);

    return instrument::Extension(node, stmt);
  }

  XPLPragma::FunctionReplacement
  XPLPragma::functionReplacement()
  {
    assert(kind == replace);

    text = ltrim(text);

    SgStatement*           marked = si::getNextStatement(&node);
    if (!marked)
      throw std::runtime_error("xpl replace not succeeded by a valid statement.");

    SgFunctionDeclaration* repl = isSgFunctionDeclaration(marked);
    if (!repl)
      throw std::runtime_error("xpl replace not succeeded by a valid function declaration.");

    return FunctionReplacement(text, repl);
  }

  const std::string XPLPragma::KERNEL_LAUNCH("kernel-launch");
}
