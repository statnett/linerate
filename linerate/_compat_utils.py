class _Annotated:
    def __getitem__(self, params):
        if isinstance(params, type):
            return params
        return params[0]


Annotated = _Annotated()
