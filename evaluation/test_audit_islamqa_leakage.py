import unittest

from audit_islamqa_leakage import audit


def question(gid, text, source, *, ayah='1:1'):
    return {'group_id': gid, 'component_id': 'c_'+gid,
            'queries': [{'query_id': gid+':q0', 'text': text}],
            'evidence': {'direct': [{'ayah': ayah, 'judgments': [{'source_id': source, 'query_id': gid+':q0'}]}]}}


def source(text, answer):
    return {'question': text, 'answer': answer, 'extracted_references': [{'text': 'verse'}]}


class LeakageAuditTests(unittest.TestCase):
    def test_distinct_queries_sharing_target_ayah_are_not_leakage(self):
        data = {'train': [question('a','first question','train:0:0')], 'validation': [], 'test': [question('b','second question','benchmark:0:0')]}
        report = audit(data, {'train': [source('first question','first answer')], 'benchmark': [source('second question','second answer')]})
        self.assertTrue(report['structural_source_checks_pass'])

    def test_different_titles_with_identical_answers_cross_split_fail(self):
        data = {'train': [question('a','long question','train:0:0')], 'validation': [], 'test': [question('b','short title','benchmark:0:0')]}
        report = audit(data, {'train': [source('long question','Same full answer.')], 'benchmark': [source('short title','Same full answer.')]})
        self.assertFalse(report['structural_source_checks_pass'])
        self.assertEqual(report['byte_identical_answer_overlap_count'],1)
        self.assertEqual(report['affected_groups_by_split'],{'test':1,'train':1})

    def test_same_family_within_train_is_allowed(self):
        data = {'train': [question('a','long question','train:0:0'),question('b','short title','benchmark:0:0')], 'validation': [], 'test': []}
        report = audit(data, {'train': [source('long question','same answer')], 'benchmark': [source('short title','same answer')]})
        self.assertTrue(report['structural_source_checks_pass'])

    def test_normalized_query_overlap_is_detected_under_distinct_ids(self):
        data = {'train': [question('a','A Question?','train:0:0')], 'validation': [question('b','a question','benchmark:0:0')], 'test': []}
        report = audit(data, {'train': [source('A Question?','first answer')], 'benchmark': [source('a question','second answer')]})
        self.assertEqual(report['cross_split_identity_counts']['normalized_question'],1)
        self.assertFalse(report['structural_source_checks_pass'])

    def test_bad_source_index_and_mismatched_source_question_fail(self):
        for sid, raw_question in [('train:4:0','query'),('train:-1:0','query'),('train:0:0','different')]:
            with self.subTest(sid=sid, raw_question=raw_question):
                report = audit({'train':[question('a','query',sid)],'validation':[],'test':[]}, {'train':[source(raw_question,'answer')], 'benchmark':[]})
                self.assertFalse(report['structural_source_checks_pass'])
                self.assertEqual(len(report['invalid_provenance']),1)

    def test_manifest_true_never_certifies_blindness(self):
        report = audit({'train':[],'validation':[],'test':[]}, {'train':[],'benchmark':[]}, {'test_set_blind':True})
        self.assertFalse(report['blindness']['certified'])
        self.assertEqual(report['blindness']['status'],'unverified')

    def test_historical_membership_is_reported_without_claiming_model_training(self):
        report = audit({'train':[],'validation':[],'test':[question('a','query','train:0:0')]}, {'train':[source('query','answer')],'benchmark':[]})
        self.assertTrue(report['structural_source_checks_pass'])
        self.assertEqual(report['legacy_source_question_matches_by_group']['train']['test'],1)


if __name__ == '__main__':
    unittest.main()
