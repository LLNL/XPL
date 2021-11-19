
  template<class _Sp>
    struct traitor
    {
    };

  template<typename _Tp>
    struct trait 
    {
    };

  template<typename _Tp>
    struct tester
    : traitor<trait<_Tp> >
    { };
