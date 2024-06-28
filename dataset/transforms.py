from jbag.transforms import Transform


class GetBoundary(Transform):
    def __init__(self, keys, boundary_dict):
        super().__init__(keys)
        self.boundary_dict = boundary_dict

    def _call_fun(self, data, *args, **kwargs):
        subject = data['subject']
        inferior, superior = self.boundary_dict[subject]
        data['inferior_slice'] = inferior
        data['superior_slice'] = superior
        for key in self.keys:
            value = data[key][..., inferior: superior + 1]
            data[key] = value
        return data
