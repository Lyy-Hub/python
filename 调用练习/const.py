import sys

class _const:
    class ConstError(TypeError):
        pass
    class ConstCaseError(ConstError):
        pass
    def __setattr__(self, name, value):
        print("name:"+name)
        print("value:"+value)
        print("==========")
        if name in self.__dict__:
            raise self.ConstError("Can't change const.%s" % name)
        if not name.isupper():
            raise self.ConstCaseError(
                "const name '%s' is not all uppercase" % name)
        self.__dict__[name] = value
    def __delattr__(self, name):
        if name in self._COMPANY_dict__:
            raise self.ConstError("can't unbind const(%s)" % name)
        raise NameError(name)

sys.modules[__name__] = _const()