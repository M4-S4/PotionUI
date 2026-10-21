import sys
import types

import pytest
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch

cv2 = pytest.importorskip("cv2", reason="cv2 required by video frame extraction", exc_type=ImportError)

from src.pipelines.pipes.video_frame_extractor.main import VideoFrameExtractorPipe
from src.pipelines.contracts import PipeInput, IOType


class TestVideoFrameExtractorPipe:

    def create_test_video(self, duration_sec: float = 5.0, fps: float = 30.0) -> str:
        """Create a test video file"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            video_path = temp_file.name

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, height = 640, 480
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        total_frames = int(duration_sec * fps)

        for i in range(total_frames):
            # Create frame with changing color (red to blue gradient)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            red_value = int(255 * (1 - i / total_frames))
            blue_value = int(255 * (i / total_frames))
            frame[:, :, 0] = blue_value  # BGR format
            frame[:, :, 2] = red_value

            # Add frame number text
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            writer.write(frame)

        writer.release()
        return video_path


    def test_extract_frames_basic(self):
        """Test basic frame extraction"""
        # Create test video
        video_path = self.create_test_video(duration_sec=2.0, fps=10.0)

        try:
            # Create pipe with basic config
            config = {"frame_rate": 2.0}  # Extract 2 frames per second
            pipe = VideoFrameExtractorPipe(config)

            # Mock generation outputs
            mock_outputs = Mock()

            # Extract frames
            frames = pipe.extract_frames_from_video(video_path, mock_outputs)

            # Should extract approximately 4 frames (2 seconds * 2 fps)
            assert len(frames) >= 3
            assert len(frames) <= 5

            # Check frames are PIL Images
            for frame in frames:
                assert isinstance(frame, Image.Image)
                assert frame.size == (640, 480)
                assert frame.mode == "RGB"

            # Verify mock was called for progress updates
            assert mock_outputs.call_count > 0

        finally:
            # Cleanup
            Path(video_path).unlink(missing_ok=True)

    def test_extract_frames_with_time_range(self):
        """Test frame extraction with time range"""
        # Create test video
        video_path = self.create_test_video(duration_sec=5.0, fps=10.0)

        try:
            # Create pipe with time range config
            config = {
                "frame_rate": 1.0,
                "start_time": 1.0,  # Start at 1 second
                "end_time": 3.0,    # End at 3 seconds
            }
            pipe = VideoFrameExtractorPipe(config)

            # Mock generation outputs
            mock_outputs = Mock()

            # Extract frames
            frames = pipe.extract_frames_from_video(video_path, mock_outputs)

            # Should extract approximately 2 frames (2 seconds * 1 fps)
            assert len(frames) >= 1
            assert len(frames) <= 3

        finally:
            # Cleanup
            Path(video_path).unlink(missing_ok=True)

    def test_extract_frames_with_max_frames(self):
        """Test frame extraction with max frames limit"""
        # Create test video
        video_path = self.create_test_video(duration_sec=5.0, fps=10.0)

        try:
            # Create pipe with max frames limit
            config = {
                "frame_rate": 5.0,   # Would normally extract many frames
                "max_frames": 3      # But limit to 3 frames
            }
            pipe = VideoFrameExtractorPipe(config)

            # Mock generation outputs
            mock_outputs = Mock()

            # Extract frames
            frames = pipe.extract_frames_from_video(video_path, mock_outputs)

            # Should extract exactly 3 frames due to limit
            assert len(frames) == 3

        finally:
            # Cleanup
            Path(video_path).unlink(missing_ok=True)

    def test_process_single_video(self):
        """Test processing a single video"""
        # Create test video
        video_path = self.create_test_video(duration_sec=1.0, fps=5.0)

        try:
            # Create pipe
            config = {"frame_rate": 1.0}
            pipe = VideoFrameExtractorPipe(config)

            # Create pipe input
            pipe_input = PipeInput(input={"video": [video_path]})

            # Mock generation outputs
            mock_outputs = Mock()

            # Process
            output = pipe.process(pipe_input, mock_outputs)

            # Check output
            assert "image" in output.output
            assert "frame_metadata" in output.output

            images = output.output["image"]
            metadata = output.output["frame_metadata"]

            assert isinstance(images, list)
            assert len(images) > 0
            assert isinstance(metadata, list)
            assert len(metadata) == 1  # One video processed

            # Check metadata structure
            video_metadata = metadata[0]
            assert "video_path" in video_metadata
            assert "video_name" in video_metadata
            assert "extracted_frames" in video_metadata
            assert "frame_rate" in video_metadata
            assert video_metadata["frame_rate"] == 1.0

        finally:
            # Cleanup
            Path(video_path).unlink(missing_ok=True)

    def test_process_multiple_videos(self):
        """Test processing multiple videos"""
        # Create two test videos
        video_path1 = self.create_test_video(duration_sec=1.0, fps=5.0)
        video_path2 = self.create_test_video(duration_sec=1.5, fps=5.0)

        try:
            # Create pipe
            config = {"frame_rate": 2.0}
            pipe = VideoFrameExtractorPipe(config)

            # Create pipe input with multiple videos
            pipe_input = PipeInput(input={"video": [video_path1, video_path2]})

            # Mock generation outputs
            mock_outputs = Mock()

            # Process
            output = pipe.process(pipe_input, mock_outputs)

            # Check output
            images = output.output["image"]
            metadata = output.output["frame_metadata"]

            assert len(metadata) == 2  # Two videos processed
            assert len(images) > 2     # Should have frames from both videos

        finally:
            # Cleanup
            Path(video_path1).unlink(missing_ok=True)
            Path(video_path2).unlink(missing_ok=True)

    def test_process_no_video_input(self):
        """Test processing with no video input"""
        # Create pipe
        pipe = VideoFrameExtractorPipe({})

        # Create pipe input with no videos
        pipe_input = PipeInput(input={})

        # Mock generation outputs
        mock_outputs = Mock()

        # Process
        output = pipe.process(pipe_input, mock_outputs)

        # Check output is empty
        assert output.output["image"] == []
        assert output.output["frame_metadata"] == []

    def test_invalid_video_path(self):
        """Test handling of invalid video path"""
        # Create pipe
        pipe = VideoFrameExtractorPipe({})

        # Mock generation outputs
        mock_outputs = Mock()

        # Try to extract from non-existent file
        frames = pipe.extract_frames_from_video("/non/existent/video.mp4", mock_outputs)

        # Should return empty list
        assert frames == []

        # Should have called mock_outputs with error
        assert mock_outputs.call_count > 0

    def test_frame_index_bypasses_fps_loop_and_uses_shared_helper(self):
        pipe = VideoFrameExtractorPipe({"frame_index": -1})
        fake_frame = Image.new("RGB", (4, 4), color=(1, 2, 3))
        outputs = []

        with patch(
            "src.pipelines.pipes._shared.media.frame_extract.extract_frame", return_value=fake_frame
        ) as mock_extract:
            frames = pipe.extract_frames_from_video("fake_video.mp4", outputs.append)

        mock_extract.assert_called_once_with("fake_video.mp4", -1)
        assert frames == [fake_frame]

    def test_frame_index_zero_is_respected_not_treated_as_falsy(self):
        pipe = VideoFrameExtractorPipe({"frame_index": 0})
        fake_frame = Image.new("RGB", (4, 4))

        with patch(
            "src.pipelines.pipes._shared.media.frame_extract.extract_frame", return_value=fake_frame
        ) as mock_extract:
            pipe.extract_frames_from_video("fake_video.mp4", lambda _o: None)

        mock_extract.assert_called_once_with("fake_video.mp4", 0)

    def test_frame_index_none_does_not_use_shared_helper(self):
        pipe = VideoFrameExtractorPipe({})

        with patch("src.pipelines.pipes._shared.media.frame_extract.extract_frame") as mock_extract:
            pipe.extract_frames_from_video("fake_video.mp4", lambda _o: None)

        mock_extract.assert_not_called()

    def test_frame_index_error_reported_and_returns_empty(self):
        pipe = VideoFrameExtractorPipe({"frame_index": 5})
        outputs = []

        with patch(
            "src.pipelines.pipes._shared.media.frame_extract.extract_frame",
            side_effect=ValueError("frame index 5 out of range"),
        ):
            frames = pipe.extract_frames_from_video("fake_video.mp4", outputs.append)

        assert frames == []

    def test_fps_loop_seek_failure_falls_back_instead_of_truncating(self):

        class FakeVideoCapture:
            def __init__(self, path):
                pass

            def isOpened(self):
                return True

            def get(self, prop):
                return {"FRAME_COUNT": 10, "FPS": 10.0}.get(prop, 0)

            def set(self, prop, val):
                pass

            def read(self):
                return False, None

            def release(self):
                pass

        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.CAP_PROP_FRAME_COUNT = "FRAME_COUNT"
        fake_cv2.CAP_PROP_FPS = "FPS"
        fake_cv2.CAP_PROP_POS_FRAMES = "POS_FRAMES"
        fake_cv2.COLOR_BGR2RGB = "BGR2RGB"
        fake_cv2.VideoCapture = FakeVideoCapture
        fake_cv2.cvtColor = lambda frame, code: frame

        fake_frame = Image.new("RGB", (4, 4))
        pipe = VideoFrameExtractorPipe({"frame_rate": 10.0})

        with patch.dict(sys.modules, {"cv2": fake_cv2}), patch(
            "src.pipelines.pipes._shared.media.frame_extract.extract_frame",
            return_value=fake_frame,
        ) as mock_extract:
            frames = pipe.extract_frames_from_video("fake_video.mp4", lambda _o: None)

        assert len(frames) == 10
        assert mock_extract.call_count == 10
