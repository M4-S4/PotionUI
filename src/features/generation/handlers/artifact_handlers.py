import logging
import os
from typing import Dict, Any

from src.pipelines.outputs import (
    CompareImagesGenerationOutput,
    ProgressGenerationOutput,
    TimerGenerationOutput,
    ModelsGenerationOutput,
    SeedGenerationOutput,
    RenderedPromptGenerationOutput,
    WarmStartGenerationOutput,
    ComfyUIWorkflowGenerationOutput,
    DiffTextGenerationOutput,
    TextGenerationOutput,
    ModelFileGenerationOutput,
)

from src.features.generation.handlers.base_handler import BaseGenerationOutputHandler
from src.features.generation.output_types import OutputTypeSpec, SerializeContext, output_type_registry
from src.features.generation.media_utils import create_base64_image

logger = logging.getLogger(__name__)


class ModelFileGenerationOutputHandler(BaseGenerationOutputHandler):
    def handle(self, output: ModelFileGenerationOutput) -> Dict[str, Any]:
        metadata = {
            'handler': 'ModelFileGenerationOutputHandler',
            'processed': True,
            'indexed': False,
        }
        if self.model_file_indexer is None:
            return metadata
        try:
            result = self.model_file_indexer(output.path)
        except Exception as e:
            logger.warning(f"Could not index model file {output.path}: {e}")
            output.index_warning = str(e)
            metadata['index_error'] = str(e)
            return metadata
        output.model_id = result.get('model_id')
        output.indexed = bool(result.get('indexed'))
        output.index_warning = result.get('warning')
        metadata['indexed'] = output.indexed
        metadata['model_id'] = output.model_id
        return metadata


def serialize_compare_images_output(output: CompareImagesGenerationOutput, ctx: SerializeContext) -> Dict[str, Any]:
    """Serialize CompareImagesGenerationOutput for pipe_artifact messages."""
    result = {
        'artifact_type': 'compare_images',
        'artifact_data': {}
    }

    try:
        compare_data = {}

        if output.compare[1] is not None:
            # Image saving is now handled by core handlers, just create base64 for display
            compare_data['compare_image'] = create_base64_image(output.compare[1])
            compare_data['compare_label'] = output.compare[0]

        if output.to[1] is not None:
            # Image saving is now handled by core handlers, just create base64 for display
            compare_data['to_image'] = create_base64_image(output.to[1])
            compare_data['to_label'] = output.to[0]

        result['artifact_data'] = compare_data
    except Exception as e:
        logger.error(f"Failed to serialize compare images: {str(e)}")
        result['artifact_data'] = {'error': str(e)}

    return result


def serialize_progress_output(output: ProgressGenerationOutput, ctx: SerializeContext) -> Dict[str, Any]:
    """Serialize ProgressGenerationOutput for generation_status messages."""
    result = {
        'status': 'running',  # Progress outputs indicate generation is running
        'current_step': getattr(output, 'state', ''),
        'message': getattr(output, 'title', ''),
    }

    # Handle Progress objects to get progress percentage. A stage that hasn't
    # reported a fraction (e.g. a cold model load with no per-component
    # breakdown) stays `None` rather than collapsing to 0.0 - a real 0% and
    # "unknown, still running" are different states, and coercing both to the
    # same number is what made the frontend bar look stuck/reset instead of
    # indeterminate (see generate/+page.svelte's indeterminate handling).
    if hasattr(output, 'progress') and output.progress is not None:
        try:
            if output.progress.max > 0:
                result['progress'] = output.progress.current / output.progress.max
            else:
                result['progress'] = 0.0
        except Exception as e:
            logger.error(f"Failed to serialize progress: {str(e)}")
            result['progress'] = None
    else:
        result['progress'] = None

    # Copy other fields if they exist
    if getattr(output, 'current_step_num', None) is not None:
        result['current_step_num'] = output.current_step_num
    if getattr(output, 'total_steps', None) is not None:
        result['total_steps'] = output.total_steps
    # Video Director: which segment/shot this progress update belongs to, for
    # a film-scoped chain/window generation (see
    # ProgressGenerationOutput.segment_id) -- lets the console light up the
    # active row of a continuous render. Omitted (not just null) for every
    # non-Director progress output.
    if getattr(output, 'segment_id', None):
        result['segment_id'] = output.segment_id
    for key in ('item_index', 'item_count', 'item_progress', 'prompt_index', 'prompt_count'):
        value = getattr(output, key, None)
        if value is not None:
            result[key] = value
    if hasattr(output, 'created_at'):
        result['created_at'] = output.created_at
    if hasattr(output, 'completed_at'):
        result['completed_at'] = output.completed_at

    return result


def serialize_seed_output(output: SeedGenerationOutput, ctx: SerializeContext) -> Dict[str, Any]:
    """Serialize SeedGenerationOutput for pipe_artifact messages."""
    return {
        'artifact_type': 'seed',
        'artifact_data': {
            'seed': output.seed
        }
    }


def serialize_rendered_prompt_output(output: RenderedPromptGenerationOutput, ctx: SerializeContext) -> Dict[str, Any]:
    """Serialize RenderedPromptGenerationOutput for pipe_artifact messages."""
    return {
        'artifact_type': 'rendered_prompt',
        'artifact_data': {
            'index': output.index,
            'positive': output.positive,
            'negative': output.negative,
        }
    }


def serialize_warm_start_output(output: WarmStartGenerationOutput, ctx: SerializeContext) -> Dict[str, Any]:
    """Serialize WarmStartGenerationOutput for pipe_artifact messages."""
    return {
        'artifact_type': 'warm_start',
        'artifact_data': {
            'resume_step': output.resume_step,
            'total_steps': output.total_steps,
            'steps_skipped': output.steps_skipped,
            'similarity': output.similarity,
        }
    }


def serialize_diff_text_output(output: DiffTextGenerationOutput, ctx: SerializeContext) -> Dict[str, Any]:
    """Serialize DiffTextGenerationOutput for pipe_artifact messages."""
    return {
        'artifact_type': 'diff_text',
        'artifact_data': {
            'name': output.name,
            'diff': output.diff,
            'negative_applied': output.negative_applied
        }
    }


def serialize_text_output(output: TextGenerationOutput, ctx: SerializeContext) -> Dict[str, Any]:
    action = output.action
    return {
        'artifact_type': 'text',
        'artifact_data': {
            'index': output.index,
            'title': output.title,
            'text': output.text,
            'mono': bool(output.mono),
            'action': None if action is None else {
                'label': action.label,
                'field': action.field,
                'values': dict(action.values or {}),
            },
        }
    }


def serialize_model_file_output(output: ModelFileGenerationOutput, ctx: SerializeContext) -> Dict[str, Any]:
    return {
        'artifact_type': 'model_file',
        'artifact_data': {
            'index': output.index,
            'kind': output.kind,
            'model_type': output.model_type,
            'file_name': output.file_name or os.path.basename(output.path),
            'size_bytes': output.size_bytes,
            'source_size_bytes': output.source_size_bytes,
            'format': output.format,
            'summary': dict(output.summary or {}),
            'model_id': output.model_id,
            'indexed': output.indexed,
            'index_warning': output.index_warning,
        }
    }


def serialize_models_output(output: ModelsGenerationOutput, ctx: SerializeContext) -> Dict[str, Any]:
    """Serialize ModelsGenerationOutput for pipe_artifact messages."""
    result = {
        'artifact_type': 'models',
        'artifact_data': {'models': []}
    }

    try:
        models_data = []
        for model in output.models:
            model_data = {
                'name': str(model.name),
                'type': str(model.type)
            }
            if hasattr(model, 'weight') and model.weight is not None:
                model_data['weight'] = float(model.weight)
            if getattr(model, 'matched_params', None) is not None:
                model_data['matched_params'] = int(model.matched_params)
            if getattr(model, 'unmatched_keys', None) is not None:
                model_data['unmatched_keys'] = int(model.unmatched_keys)
            if getattr(model, 'zero_effect', None) is not None:
                model_data['zero_effect'] = bool(model.zero_effect)
            if getattr(model, 'ignored', None):
                model_data['ignored'] = [str(item) for item in model.ignored]
            if getattr(model, 'source_id', None):
                model_data['source_id'] = str(model.source_id)
            if getattr(model, 'unmatched_sample', None):
                model_data['unmatched_sample'] = [str(item) for item in model.unmatched_sample]
            if getattr(model, 'reason', None):
                model_data['reason'] = str(model.reason)
            models_data.append(model_data)

        result['artifact_data']['models'] = models_data
    except Exception as e:
        logger.error(f"Failed to serialize models: {str(e)}")
        result['artifact_data'] = {
            'models': [],
            'error': str(e)
        }

    return result


def serialize_timer_output(output: TimerGenerationOutput, ctx: SerializeContext) -> Dict[str, Any]:
    """Serialize TimerGenerationOutput for timer_update messages."""
    return {
        'timer_name': output.name,
        'timer_value': output.value,
        'timer_unit': output.unit,
        'formatted_time': f"{output.value:.2f}{output.unit}"
    }


def serialize_workflow_output(output: ComfyUIWorkflowGenerationOutput, ctx: SerializeContext) -> Dict[str, Any]:
    """Serialize ComfyUIWorkflowGenerationOutput for pipe_artifact messages."""
    return {
        'artifact_type': 'workflow',
        'artifact_data': {
            'workflow': output.workflow,
            'node_count': output.node_count,
            'workflow_file': output.workflow_file
        }
    }


output_type_registry.register(OutputTypeSpec(
    output_cls=CompareImagesGenerationOutput,
    key='compare_images',
    message_type='pipe_artifact',
    serializer=serialize_compare_images_output,
))

output_type_registry.register(OutputTypeSpec(
    output_cls=ProgressGenerationOutput,
    key='progress',
    message_type='generation_status',
    serializer=serialize_progress_output,
))

output_type_registry.register(OutputTypeSpec(
    output_cls=TimerGenerationOutput,
    key='timer',
    message_type='timer_update',
    serializer=serialize_timer_output,
))

output_type_registry.register(OutputTypeSpec(
    output_cls=ModelsGenerationOutput,
    key='models',
    message_type='pipe_artifact',
    serializer=serialize_models_output,
))

output_type_registry.register(OutputTypeSpec(
    output_cls=SeedGenerationOutput,
    key='seed',
    message_type='pipe_artifact',
    serializer=serialize_seed_output,
))

output_type_registry.register(OutputTypeSpec(
    output_cls=RenderedPromptGenerationOutput,
    key='rendered_prompt',
    message_type='pipe_artifact',
    serializer=serialize_rendered_prompt_output,
))

output_type_registry.register(OutputTypeSpec(
    output_cls=WarmStartGenerationOutput,
    key='warm_start',
    message_type='pipe_artifact',
    serializer=serialize_warm_start_output,
))

output_type_registry.register(OutputTypeSpec(
    output_cls=ComfyUIWorkflowGenerationOutput,
    key='comfyui_workflow',
    message_type='pipe_artifact',
    serializer=serialize_workflow_output,
))

output_type_registry.register(OutputTypeSpec(
    output_cls=DiffTextGenerationOutput,
    key='diff_text',
    message_type='pipe_artifact',
    serializer=serialize_diff_text_output,
))

output_type_registry.register(OutputTypeSpec(
    output_cls=TextGenerationOutput,
    key='text',
    message_type='pipe_artifact',
    serializer=serialize_text_output,
))

output_type_registry.register(OutputTypeSpec(
    output_cls=ModelFileGenerationOutput,
    key='model_file',
    message_type='pipe_artifact',
    serializer=serialize_model_file_output,
    handler_cls=ModelFileGenerationOutputHandler,
))
