import logging
import yaml
from typing import Dict, List, Any, Optional, Tuple
from src.platform.util.ids import generate_ulid
from src.features.phrasebook.dto import PhrasebookCategory, PhrasebookValue
from src.features.phrasebook.repository import (
    PhrasebookCategoryRepository,
    PhrasebookValueRepository,
    phrasebook_category_repo,
    phrasebook_value_repo
)
from src.features.phrasebook.hooks import PHRASEBOOK_HOOKS
from src.platform.filesystem.yaml_io import load_yaml

logger = logging.getLogger(__name__)


class PhrasebookImportManager:
    def __init__(
        self,
        category_repository: PhrasebookCategoryRepository = phrasebook_category_repo,
        value_repository: PhrasebookValueRepository = phrasebook_value_repo,
    ):
        self.category_repo = category_repository
        self.value_repo = value_repository

    def _get_plugin_registry(self):
        from src.platform.plugins.runtime_registries import get_global_plugin_registry
        return get_global_plugin_registry()

    def _execute_hook(self, hook: str, data: dict) -> Tuple[dict, bool]:
        plugins = self._get_plugin_registry()
        context, results = plugins.execute_hook(
            hook,
            initial_data=data
        )


        blocked = context.data.get("blocked", False)

        return context.data, blocked

    def import_yaml(
        self, content: str, user_id: str, root_category: Optional[str] = None
    ) -> Dict[str, Any]:
        yaml_content = content
        try:

            hook_data, blocked = self._execute_hook(
                PHRASEBOOK_HOOKS.before_import,
                {
                    "yaml_content": yaml_content,
                    "user_id": user_id,
                    "root_category": root_category
                }
            )

            if blocked:
                reason = hook_data.get("block_reason", "Import blocked by plugin")
                logger.warning(f"Phrasebook import blocked by plugin: {reason}")
                return {
                    'success': False,
                    'error': reason,
                    'categories_created': 0,
                    'values_created': 0
                }


            yaml_content = hook_data.get("yaml_content", yaml_content)
            root_category = hook_data.get("root_category", root_category)

            data = load_yaml(yaml_content)

            if data is None:
                return {
                    'success': False,
                    'error': 'Empty YAML file',
                    'categories_created': 0,
                    'values_created': 0
                }


            if isinstance(data, dict):

                result = self._import_nested_dict(data, user_id, root_category)
            elif isinstance(data, list):

                result = self._import_label_value_list(data, user_id, root_category)
            else:
                return {
                    'success': False,
                    'error': 'Unsupported YAML format. Expected dict or list.',
                    'categories_created': 0,
                    'values_created': 0
                }


            if result.get('success'):
                self._execute_hook(
                    PHRASEBOOK_HOOKS.after_import,
                    {
                        "user_id": user_id,
                        "root_category": root_category,
                        "categories_created": result.get('categories_created', 0),
                        "values_created": result.get('values_created', 0)
                    }
                )
                logger.info(
                    f"Phrasebook import completed: {result.get('categories_created', 0)} categories, "
                    f"{result.get('values_created', 0)} values"
                )

            return result

        except yaml.YAMLError as e:
            return {
                'success': False,
                'error': f'Invalid YAML format: {str(e)}',
                'categories_created': 0,
                'values_created': 0
            }
        except Exception as e:
            logger.error(f"Phrasebook import failed: {e}")
            return {
                'success': False,
                'error': f'Import failed: {str(e)}',
                'categories_created': 0,
                'values_created': 0
            }

    def _import_nested_dict(self, data: Dict, user_id: str, root_category: Optional[str] = None) -> Dict[str, Any]:
        categories_created = 0
        values_created = 0


        if root_category:
            data = {root_category: data}


        for key, value in data.items():
            cat_count, val_count = self._process_nested_structure(
                key, value, user_id, parent_id=None, parent_path=""
            )
            categories_created += cat_count
            values_created += val_count

        return {
            'success': True,
            'categories_created': categories_created,
            'values_created': values_created
        }

    def _import_label_value_list(self, data: List, user_id: str, root_category: Optional[str] = None) -> Dict[str, Any]:
        if not root_category:
            root_category = 'imported'


        category = self._ensure_category(root_category, root_category, user_id, None)
        if not category:
            return {
                'success': False,
                'error': 'Failed to create root category',
                'categories_created': 0,
                'values_created': 0
            }


        values = []
        for idx, item in enumerate(data):
            if isinstance(item, dict):

                label = item.get('label', '')
                value = item.get('value', '')
                if not label:
                    label = value
            elif isinstance(item, str):

                label = item
                value = item
            else:
                continue

            if label and value:
                values.append(PhrasebookValue(
                    id=generate_ulid(),
                    category_id=category.id,
                    label=label,
                    value=value,
                    sort_order=idx,
                    user_id=user_id
                ))


        values_created = self.value_repo.create_bulk(values)

        return {
            'success': True,
            'categories_created': 1,
            'values_created': values_created
        }

    def _process_nested_structure(
        self,
        name: str,
        data: Any,
        user_id: str,
        parent_id: Optional[str],
        parent_path: str
    ) -> Tuple[int, int]:
        categories_created = 0
        values_created = 0


        current_path = f"{parent_path}.{name}" if parent_path else name


        category = self._ensure_category(name, current_path, user_id, parent_id)
        if not category:
            return 0, 0

        categories_created += 1

        if isinstance(data, dict):

            for key, value in data.items():
                cat_count, val_count = self._process_nested_structure(
                    key, value, user_id, category.id, current_path
                )
                categories_created += cat_count
                values_created += val_count

        elif isinstance(data, list):

            values = []
            for idx, item in enumerate(data):
                if isinstance(item, dict):

                    label = item.get('label', '')
                    value = item.get('value', '')
                    if not label:
                        label = value
                elif isinstance(item, str):

                    label = item
                    value = item
                else:
                    continue

                if label and value:
                    values.append(PhrasebookValue(
                        id=generate_ulid(),
                        category_id=category.id,
                        label=label,
                        value=value,
                        sort_order=idx,
                        user_id=user_id
                    ))

            if values:
                values_created += self.value_repo.create_bulk(values)

        return categories_created, values_created

    def _ensure_category(
        self,
        name: str,
        path: str,
        user_id: str,
        parent_id: Optional[str]
    ) -> Optional[PhrasebookCategory]:

        existing = self.category_repo.get_by_path(path, user_id)
        if existing:
            return existing


        category = PhrasebookCategory(
            id=generate_ulid(),
            name=name,
            path=path,
            parent_id=parent_id,
            user_id=user_id,
            description=f"Auto-imported category: {name}"
        )

        success = self.category_repo.create(category)
        return category if success else None

    def export_to_yaml(self, category_id: str, user_id: str) -> Optional[str]:
        category = self.category_repo.get_by_id(category_id, user_id)
        if not category:
            return None


        children = self.category_repo.get_children(category_id, user_id)

        if children:

            result = self._export_nested_structure(category, user_id)
        else:

            values = self.value_repo.get_by_category(category_id, user_id)
            if not values:
                return "[]"

            result = []
            for value in values:
                if value.label == value.value:

                    result.append(value.value)
                else:

                    result.append({
                        'label': value.label,
                        'value': value.value
                    })

        return yaml.dump(result, default_flow_style=False, allow_unicode=True)

    def _export_nested_structure(self, category: PhrasebookCategory, user_id: str) -> Dict:
        result = {}


        values = self.value_repo.get_by_category(category.id, user_id)
        if values:
            value_list = []
            for value in values:
                if value.label == value.value:
                    value_list.append(value.value)
                else:
                    value_list.append({
                        'label': value.label,
                        'value': value.value
                    })
            if value_list:
                result = value_list


        children = self.category_repo.get_children(category.id, user_id)
        for child in children:
            child_data = self._export_nested_structure(child, user_id)
            if isinstance(result, list):

                result = {category.name: result}
            result[child.name] = child_data

        return result


phrasebook_import_manager = PhrasebookImportManager()
