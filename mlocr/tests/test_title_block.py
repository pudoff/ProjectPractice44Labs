"""Regression tests for author-row detection and role assignment in the title block."""
import unittest

from mlocr.src.normalization import enrich
from mlocr.src.pdf_text import build_title_block
from mlocr.src.title_block import match_role, merge_title_block, title_block_from_cells


def cell_text(id_, text, x, y):
    """A text at x, y in mm inside the title block of an A3 sheet (420x297, y up, px == mm)."""
    X, Y = 420 - 5 - 185 + x, 5 + y
    return enrich({'id': id_, 'text': text, 'page': 'page-1',
                   'poly_source_px': [[X - 8, Y - 2], [X + 8, Y - 2], [X + 8, Y + 2], [X - 8, Y + 2]]})


def sheet_cv():
    return {'scale_status': 'known', 'calibration': {'sheet_mm': [420, 297]}}


class AuthorRowTests(unittest.TestCase):
    def test_roles_come_from_printed_labels_not_from_row_position(self):
        texts = [cell_text('a1', 'Разраб.', 8, 27.5), cell_text('n1', 'Фомичев', 28, 27.5),
                 cell_text('a2', 'Пров.', 8, 22.5), cell_text('n2', 'Крюков', 28, 22.5),
                 cell_text('a3', 'ГНП', 8, 17.5), cell_text('n3', 'Шурухин', 28, 17.5),
                 cell_text('a4', 'H.контр.', 8, 12.5), cell_text('n4', 'Васильева', 28, 12.5),
                 cell_text('a5', 'Haч. omdena', 8, 7.5), cell_text('a5b', 'отдела', 13, 7.5),
                 cell_text('n5', 'Вишняков', 28, 7.5)]
        fields = merge_title_block({}, {'texts': texts}, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], sheet_cv())['page-1']['fields']
        self.assertEqual(fields['authors']['value'], [
            {'role': 'Разраб.', 'name': 'Фомичев'},
            {'role': 'Пров.', 'name': 'Крюков'},
            {'role': 'ГИП', 'name': 'Шурухин'},
            {'role': 'Н.контр.', 'name': 'Васильева'},
            {'role': 'Нач. отдела', 'name': 'Вишняков'},
        ])

    def test_placeholder_is_not_a_person_and_empty_rows_stay_unknown(self):
        texts = [cell_text('a1', 'Разраб.', 8, 27.5), cell_text('n1', 'xxxxxxx', 28, 27.5),
                 cell_text('a2', 'Пров.', 8, 22.5), cell_text('d2', 'xx.xx', 60, 22.5)]
        fields = merge_title_block({}, {'texts': texts}, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], sheet_cv())['page-1']['fields']
        self.assertEqual(fields['authors']['status'], 'unknown')

    def test_part_name_in_the_next_cell_is_not_a_surname(self):
        # On wide forms the name cell shares the row with the Разраб. label; the surname
        # column is the narrow adjacent cell only.
        texts = [cell_text('a1', 'Разраб.', 8, 27.5), cell_text('name', 'Втулка', 100, 27.5)]
        fields = merge_title_block({}, {'texts': texts}, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], sheet_cv())['page-1']['fields']
        self.assertEqual(fields['authors']['status'], 'unknown')

    def test_layout_path_anchors_on_labels_anywhere_in_the_bottom_band(self):
        items = [enrich({'id': 'label', 'text': 'Разраб.', 'poly_source_px': [[550, 3600], [600, 3600], [600, 3620], [550, 3620]]}),
                 enrich({'id': 'name', 'text': 'Фомичев', 'poly_source_px': [[605, 3600], [680, 3600], [680, 3620], [605, 3620]]}),
                 enrich({'id': 'part', 'text': 'Шайба', 'poly_source_px': [[1200, 3500], [1600, 3500], [1600, 3700], [1200, 3700]]})]
        page = {'page': 'page-1', 'width_px': 2000, 'height_px': 4000, 'texts': items}
        authors = build_title_block([page])['page-1']['fields']['authors']
        self.assertEqual(authors['value'], [{'role': 'Разраб.', 'name': 'Фомичев'}])

    def test_match_role_folds_recogniser_lookalikes(self):
        self.assertEqual(match_role('H.контр.'), 'Н.контр.')
        self.assertEqual(match_role('ГНП'), 'ГИП')
        self.assertEqual(match_role('Haч. omdena отдела'), 'Нач. отдела')
        self.assertIsNone(match_role('Масштаб'))
        self.assertIsNone(match_role('Лист'))


if __name__ == '__main__':
    unittest.main()
