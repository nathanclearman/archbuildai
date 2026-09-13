"""
Add-on preferences: environment status, one-click dependency install, base
model download, and optional overrides (VLM interpreter, weights dir).

Long-running work (pip, 15 GB download, interpreter probes) runs in a
background thread; only the modal timer on the main thread touches bpy.
"""

import os
import threading

import bpy

from .api import local_model, qwen_client, setup_env

# Shared between the worker threads and the UI. Only strings/bools/lists —
# never bpy objects — cross the thread boundary.
ENV = {
    "status": None,          # dict from local_model.environment_status()
    "cv_missing": None,      # list[str] of pip requirements missing in Blender's python
    "vlm_missing": None,
    "busy": "",              # label of the running job, or ""
    "log": [],               # last lines of the running/finished job
    "checked": False,
}
_LOG_MAX = 12


def _log(line: str) -> None:
    ENV["log"].append(line)
    del ENV["log"][:-_LOG_MAX]


def refresh_environment_blocking() -> None:
    """Probe interpreters/packages/model cache. Seconds; run in a thread."""
    py = setup_env.blender_python()
    ENV["cv_missing"] = setup_env.missing_packages(py, setup_env.CV_PACKAGES)
    ENV["vlm_missing"] = setup_env.missing_packages(py, setup_env.vlm_packages(local_model.preferred_backend()))
    ENV["status"] = local_model.environment_status()
    ENV["checked"] = True


def refresh_environment_async() -> None:
    if ENV["busy"]:
        return
    threading.Thread(target=refresh_environment_blocking, daemon=True).start()


def _prefs(context=None):
    context = context or bpy.context
    try:
        return context.preferences.addons[__package__].preferences
    except (KeyError, AttributeError):
        return None


def apply_overrides(prefs=None) -> None:
    """Push the preference overrides into the env vars local_model reads,
    then make it re-resolve. Also stops a running daemon: it may be bound
    to the old interpreter / weights."""
    prefs = prefs or _prefs()
    if prefs is None:
        return
    for key, value in (("FP3D_PYTHON", prefs.vlm_python), ("FP3D_WEIGHTS_DIR", prefs.weights_dir)):
        value = bpy.path.abspath(value).strip() if value else ""
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    qwen_client.shutdown()
    local_model.reconfigure()


def _on_override_changed(self, context):
    apply_overrides(self)
    refresh_environment_async()


class FP3D_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    vlm_python: bpy.props.StringProperty(
        name="VLM Python (optional)",
        description="Interpreter with torch + transformers + peft for the Qwen label pass. "
                    "Leave empty to auto-detect (or install into Blender's Python below)",
        subtype='FILE_PATH', default="", update=_on_override_changed,
    )
    weights_dir: bpy.props.StringProperty(
        name="Qwen weights dir (optional)",
        description="Folder with train_config.json + adapter/ for the 'Qwen2.5-VL (trained)' "
                    "backend. Not needed for Hybrid, which uses the base model only",
        subtype='DIR_PATH', default="", update=_on_override_changed,
    )

    def draw(self, context):
        layout = self.layout
        st = ENV["status"] or {}

        box = layout.box()
        box.label(text="Environment", icon='SYSTEM')
        if not ENV["checked"]:
            box.label(text="Not checked yet.", icon='QUESTION')
        else:
            cv_ok = not ENV["cv_missing"]
            row = box.row()
            row.label(text="Geometry (YOLO) packages in Blender's Python: "
                           + ("installed" if cv_ok else f"missing {', '.join(ENV['cv_missing'])}"),
                      icon='CHECKMARK' if cv_ok else 'ERROR')
            vlm_py = st.get("python")
            row = box.row()
            row.label(text=("Room-label (Qwen) interpreter: " + vlm_py) if vlm_py
                      else "Room-label (Qwen) interpreter: none found",
                      icon='CHECKMARK' if vlm_py else 'ERROR')
            row = box.row()
            row.label(text=f"Qwen2.5-VL base model ({st.get('backend', 'torch')}, "
                           f"{setup_env.BASE_MODEL_SIZE_GB.get(st.get('backend'), 15)} GB): "
                           + ("downloaded" if st.get("base_model_cached") else "not downloaded"),
                      icon='CHECKMARK' if st.get("base_model_cached") else 'ERROR')
            row = box.row()
            row.label(text="Fine-tuned adapter (optional, Qwen-only backend): "
                           + ("found" if st.get("adapter") else "not found"),
                      icon='CHECKMARK' if st.get("adapter") else 'INFO')
            if not st.get("inference_script"):
                box.label(text=f"VLM runtime missing at {st.get('model_dir')}", icon='ERROR')

        row = box.row(align=True)
        row.enabled = not ENV["busy"]
        row.operator("fp3d.env_refresh", icon='FILE_REFRESH', text="Re-check")
        op = row.operator("fp3d.env_install", icon='IMPORT', text="Install geometry packages")
        op.which = 'CV'
        op = row.operator("fp3d.env_install", icon='IMPORT', text="Install VLM packages")
        op.which = 'VLM'
        row = box.row(align=True)
        row.enabled = not ENV["busy"]
        size = setup_env.BASE_MODEL_SIZE_GB.get(st.get("backend"), 15)
        row.operator("fp3d.env_download_model", icon='URL', text=f"Download base model ({size} GB)")
        if ENV["busy"]:
            box.label(text=f"Working: {ENV['busy']}…", icon='TIME')
        if ENV["log"]:
            col = box.column(align=True)
            col.scale_y = 0.7
            for line in ENV["log"][-6:]:
                col.label(text=line[:110])

        box = layout.box()
        box.label(text="Overrides (advanced)", icon='PREFERENCES')
        box.prop(self, "vlm_python")
        box.prop(self, "weights_dir")


class _EnvJobMixin:
    """Modal operator scaffold: run `self._job()` in a thread, redraw until done."""
    _thread = None
    _timer = None

    def _start(self, context, label):
        if ENV["busy"]:
            self.report({'WARNING'}, f"Already working: {ENV['busy']}")
            return {'CANCELLED'}
        ENV["busy"] = label
        ENV["log"] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _run(self):
        try:
            self._job()
        except Exception as e:  # noqa: BLE001 — surfaced in the log
            _log(f"error: {e}")
        finally:
            try:
                refresh_environment_blocking()
            except Exception as e:  # noqa: BLE001
                _log(f"re-check failed: {e}")
            ENV["busy"] = ""

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        if self._thread and self._thread.is_alive():
            return {'PASS_THROUGH'}
        context.window_manager.event_timer_remove(self._timer)
        self.report({'INFO'}, ENV["log"][-1] if ENV["log"] else "done")
        return {'FINISHED'}


class FP3D_OT_EnvRefresh(bpy.types.Operator, _EnvJobMixin):
    bl_idname = "fp3d.env_refresh"
    bl_label = "Re-check environment"
    bl_description = "Probe Blender's Python, the VLM interpreter and the model cache"

    def _job(self):
        _log("checking…")

    def execute(self, context):
        return self._start(context, "checking environment")


class FP3D_OT_EnvInstall(bpy.types.Operator, _EnvJobMixin):
    bl_idname = "fp3d.env_install"
    bl_label = "Install packages"
    bl_description = "pip-install the missing packages into Blender's Python (user site-packages)"

    which: bpy.props.EnumProperty(items=[('CV', "Geometry", ""), ('VLM', "VLM", "")], default='CV')

    def _job(self):
        pkgs = setup_env.CV_PACKAGES if self.which == 'CV' else setup_env.vlm_packages(local_model.preferred_backend())
        code = setup_env.install_packages(setup_env.blender_python(), pkgs, _log)
        _log("install finished OK" if code == 0 else f"pip exited with code {code}")

    def execute(self, context):
        return self._start(context, f"installing {self.which} packages")


class FP3D_OT_EnvDownloadModel(bpy.types.Operator, _EnvJobMixin):
    bl_idname = "fp3d.env_download_model"
    bl_label = "Download base model"
    bl_description = "Download Qwen2.5-VL-7B-Instruct into the Hugging Face cache (~15 GB)"

    def _job(self):
        st = ENV["status"] or local_model.environment_status()
        py = st.get("python") or setup_env.blender_python()
        code = setup_env.download_base_model(py, _log, model_id=st.get("base_model", setup_env.DEFAULT_BASE_MODEL))
        _log("download finished OK" if code == 0 else f"download exited with code {code}")

    def execute(self, context):
        return self._start(context, "downloading base model")


classes = [FP3D_AddonPreferences, FP3D_OT_EnvRefresh, FP3D_OT_EnvInstall, FP3D_OT_EnvDownloadModel]
