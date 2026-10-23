import io
import sys
from unittest.mock import patch

import pytest
import yaml

import tests.conftest as ct
from src.features.phrasebook.import_manager import PhrasebookImportManager
from src.features.phrasebook.repository import (
    phrasebook_category_repo,
    phrasebook_value_repo
)


class TestPhrasebookImportManager:
    
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.platform.util.ids import generate_ulid
        import bcrypt





        test_database = ct.TestDatabase()

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:



            with patch('src.platform.database.database.db', test_database), \
                 patch('src.platform.database.migration_runner.db', test_database):
                from src.platform.database.migration_runner import MigrationRunner
                MigrationRunner().run_migrations()
        finally:
            sys.stdout = old_stdout

        with patch('src.platform.database.database.db', test_database):

            self.test_user = 'test_user_' + generate_ulid()[:8]
            with test_database.get_cursor() as cursor:

                hashed_password = bcrypt.hashpw('test123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                cursor.execute("""
                    INSERT INTO users (id, username, email, password_hash, account_type, created_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                """, (self.test_user, 'testuser', 'test@example.com', hashed_password, 'USER'))

            self.service = PhrasebookImportManager()

            yield


            with test_database.get_cursor() as cursor:
                cursor.execute("DELETE FROM phrasebook_values WHERE user_id = ?", (self.test_user,))
                cursor.execute("DELETE FROM phrasebook_categories WHERE user_id = ?", (self.test_user,))
                cursor.execute("DELETE FROM users WHERE id = ?", (self.test_user,))

        test_database.close()
    
    def test_import_nested_dict_format(self):
        yaml_content = """
clothes:
  full:
    traditional:
      - kimono
      - hanfu
      - habit
    modern:
      - suit
      - dress
  accessories:
    - hat
    - scarf
"""
        
        result = self.service.import_yaml(yaml_content, self.test_user)
        
        assert result['success'] is True
        assert result['categories_created'] == 5
        assert result['values_created'] == 7
        

        categories = phrasebook_category_repo.get_all(self.test_user)
        assert len(categories) == 5
        

        paths = [cat.path for cat in categories]
        assert 'clothes' in paths
        assert 'clothes.full' in paths
        assert 'clothes.full.traditional' in paths
        assert 'clothes.full.modern' in paths
        assert 'clothes.accessories' in paths
        

        traditional_cat = next(c for c in categories if c.path == 'clothes.full.traditional')
        values = phrasebook_value_repo.get_by_category(traditional_cat.id, self.test_user)
        assert len(values) == 3
        value_labels = [v.label for v in values]
        assert 'kimono' in value_labels
        assert 'hanfu' in value_labels
        assert 'habit' in value_labels
    
    def test_import_label_value_list_format(self):
        yaml_content = """
- label: "Happy"
  value: "happy, ^o^, open mouth, xd"
- label: "Sad"
  value: "crying, tears, sad"
- label: "Angry"
  value: "angry, rage, furious"
"""
        
        result = self.service.import_yaml(yaml_content, self.test_user, root_category='emotions')
        
        assert result['success'] is True
        assert result['categories_created'] == 1
        assert result['values_created'] == 3
        

        categories = phrasebook_category_repo.get_all(self.test_user)
        assert len(categories) == 1
        assert categories[0].path == 'emotions'
        

        values = phrasebook_value_repo.get_by_category(categories[0].id, self.test_user)
        assert len(values) == 3
        
        happy_value = next(v for v in values if v.label == 'Happy')
        assert happy_value.value == 'happy, ^o^, open mouth, xd'
        
        sad_value = next(v for v in values if v.label == 'Sad')
        assert sad_value.value == 'crying, tears, sad'
    
    def test_import_simple_list_format(self):
        yaml_content = """
- happy
- sad
- angry
- excited
"""
        
        result = self.service.import_yaml(yaml_content, self.test_user, root_category='moods')
        
        assert result['success'] is True
        assert result['categories_created'] == 1
        assert result['values_created'] == 4
        

        categories = phrasebook_category_repo.get_all(self.test_user)
        values = phrasebook_value_repo.get_by_category(categories[0].id, self.test_user)
        
        for value in values:
            assert value.label == value.value
    
    def test_import_mixed_nested_format(self):
        yaml_content = """
styles:
  art:
    - watercolor
    - oil painting
    - sketch
  photo:
    realistic:
      - portrait
      - landscape
    artistic:
      - black and white
      - vintage
"""
        
        result = self.service.import_yaml(yaml_content, self.test_user)
        
        assert result['success'] is True
        assert result['categories_created'] == 5
        assert result['values_created'] == 7
    
    def test_export_to_yaml(self):

        yaml_content = """
test:
  subcategory:
    - value1
    - value2
"""
        self.service.import_yaml(yaml_content, self.test_user)
        

        categories = phrasebook_category_repo.get_all(self.test_user)
        root_category = next(c for c in categories if c.path == 'test')
        

        exported_yaml = self.service.export_to_yaml(root_category.id, self.test_user)
        
        assert exported_yaml is not None
        

        exported_data = yaml.safe_load(exported_yaml)
        

        assert 'subcategory' in exported_data
        assert isinstance(exported_data['subcategory'], list)
        assert 'value1' in exported_data['subcategory']
        assert 'value2' in exported_data['subcategory']
    
    def test_import_empty_yaml(self):
        yaml_content = ""
        
        result = self.service.import_yaml(yaml_content, self.test_user)
        
        assert result['success'] is False
        assert 'Empty' in result['error']
    
    def test_import_invalid_yaml(self):
        yaml_content = """
this is not: valid yaml {
  because: it has: multiple: colons
"""
        
        result = self.service.import_yaml(yaml_content, self.test_user)
        
        assert result['success'] is False
        assert 'Invalid YAML' in result['error']
    
    def test_duplicate_category_handling(self):
        yaml_content = """
test:
  subcategory:
    - value1
"""
        

        result1 = self.service.import_yaml(yaml_content, self.test_user)
        result2 = self.service.import_yaml(yaml_content, self.test_user)
        

        assert result1['categories_created'] == 2
        assert result2['categories_created'] == 2
        

        categories = phrasebook_category_repo.get_all(self.test_user)
        paths = [cat.path for cat in categories]
        assert paths.count('test') == 1
        assert paths.count('test.subcategory') == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])