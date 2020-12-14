
class RectifyText:
    def __init__(
        self,
        output_process=False,
    ):
        self.output_process = output_process

    def __call__(self, field_text_location_dict, document_type,debug=False):

        self._field_text_location_dict = field_text_location_dict
        self._document_type = document_type
        self.rectified_text = self.cleanup_text(self._field_text_location_dict)
        return self.rectified_text

    def cleanup_text(self, field_text_location_dict):
        return field_text_location_dict


