"""Explicit native concentration requests in the ordinary Studio workspace."""
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QAbstractItemView)


class NativeConcentrationControls(QWidget):
    calculate_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.catalog = None
        self.frozen_path = None
        self.overflow_paths = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout(); layout.addLayout(bar)
        self.mode = QComboBox()
        self.mode.addItem('Apply registered native calibration — no refit', 'apply')
        self.mode.addItem('Fit shared native response scales to injected mass', 'series')
        self.mode.addItem('Fit individual native response to injected mass', 'individual')
        self.mode.addItem('Apply saved native fit — no refit', 'frozen')
        bar.addWidget(self.mode, 1)
        self.load_frozen = QPushButton('Choose saved native fit')
        self.load_frozen.clicked.connect(self.choose_frozen); bar.addWidget(self.load_frozen)
        self.calculate = QPushButton('Calculate concentration from selected phase images')
        self.calculate.setEnabled(False)
        self.calculate.clicked.connect(self.calculate_requested.emit); bar.addWidget(self.calculate)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Use', 'Experiment', 'Chemistry group', 'Native calibration / geometry', 'Overflow targets'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(190); layout.addWidget(self.table)
        refs = QHBoxLayout(); layout.addLayout(refs)
        self.overflow_mode = QComboBox()
        self.overflow_mode.addItem('Without overflow', 'off')
        self.overflow_mode.addItem('Compare with measured overflow gas targets', 'compare')
        self.overflow_mode.addItem('Include measured overflow gas targets in fit', 'constrain')
        refs.addWidget(self.overflow_mode)
        attach = QPushButton('Attach reviewed targets to selected experiment')
        attach.clicked.connect(self.choose_overflow); refs.addWidget(attach)
        clear = QPushButton('Remove targets')
        clear.clicked.connect(self.clear_overflow); refs.addWidget(clear)
        self.method_summary = QLabel()
        self.method_summary.setWordWrap(True)
        layout.addWidget(self.method_summary)
        self.note = QLabel('Load a phase model. Discovery reads metadata only; calculation starts on command.')
        self.note.setWordWrap(True); layout.addWidget(self.note)
        self.mode.currentIndexChanged.connect(self.update_note)
        self.overflow_mode.currentIndexChanged.connect(self.update_note)
        self.table.itemChanged.connect(self.update_note)
        self.update_note()

    def set_catalog(self, catalog):
        self.catalog = catalog
        jobs = catalog.get('jobs', []) if catalog else []
        self.table.blockSignals(True)
        self.table.setRowCount(len(jobs))
        selected = False
        for row, job in enumerate(jobs):
            config = job.get('concentration', {})
            ready = config.get('ready', False)
            group = config.get('chemistry_group', 'unconfirmed')
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsEnabled | (Qt.ItemIsUserCheckable if ready else Qt.NoItemFlags))
            use = ready and not selected
            check.setCheckState(Qt.Checked if use else Qt.Unchecked)
            selected |= use
            self.table.setItem(row, 0, check)
            reasons = config.get('reasons', [])
            status = ('Provisional imported calibration' if config.get('provisional', True) else 'Registered calibration') if ready else '; '.join(reasons) or 'Configuration not ready'
            for column, text in enumerate((job['run'].upper(), group, status, self.overflow_paths.get(job['run'], 'None')), 1):
                item = QTableWidgetItem(text); item.setToolTip(text); self.table.setItem(row, column, item)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(3, 470)
        self.calculate.setEnabled(any(j.get('concentration', {}).get('ready') for j in jobs))
        self.update_note()

    def update_note(self):
        mode = self.mode.currentData()
        selected = []
        if self.catalog:
            selected = [job for row, job in enumerate(self.catalog.get('jobs', []))
                if self.table.item(row, 0) is not None
                and self.table.item(row, 0).checkState() == Qt.Checked]
        names = ', '.join(job['run'].upper() for job in selected) or 'none'
        group = sorted({str(job.get('concentration', {}).get('chemistry_group', 'unconfirmed'))
                        for job in selected})
        scope = {'apply': 'registered native calibration; no fit',
                 'series': 'shared fit to injected mass',
                 'individual': 'individual fit to injected mass',
                 'frozen': 'saved fit; no refit'}[mode]
        overflow = {'off': 'not used',
                    'compare': 'comparison only; excluded from fit',
                    'constrain': 'used as a fit constraint at matched measured times'}[
                        self.overflow_mode.currentData()]
        target_count = sum(job['run'] in self.overflow_paths for job in selected)
        model = self.catalog.get('model_id', 'none loaded') if self.catalog else 'none loaded'
        self.method_summary.setText(
            f'Selected phase model: {model} · experiments: {names} · chemistry: '
            f'{", ".join(group) or "unconfirmed"}\n'
            f'Optical input: corrected photographs and matching baseline pinned to this phase model. '
            f'An alternative illumination or ColorChecker correction requires a separately prepared, '
            f'provenance-linked result; this native calculation does not switch corrections.\n'
            f'Gas depth: registered native gas response on the fixed 2D phase map. '
            f'Persistent full-depth gas is not part of this native calculation.\n'
            f'Calibration: {scope} · overflow: {overflow} · attached selected runs: {target_count}.')
        provenance = [f"Phase model root: {self.catalog.get('model_root', '')}" if self.catalog
                      else 'No phase model loaded']
        for job in selected:
            candidate = job.get('candidate') or {}
            registration = job.get('concentration') or {}
            provenance.append(
                f"{job['run'].upper()}: candidate {candidate.get('candidate_id', 'not recorded')}; "
                f"phase pipeline {candidate.get('params_sha256', 'not recorded')}; "
                f"optical model {candidate.get('optical_model_sha256', 'not recorded')}; "
                f"concentration profile {registration.get('profile_path', 'not recorded')}")
        self.method_summary.setToolTip('\n'.join(provenance))
        if mode in ('series', 'individual'):
            text = ('Mass is fitted to the injected-mass curve: this is calibration, not independent mass validation. '
                    'Shared scales are applied to each experiment’s registered native color paths. '
                    'Shared fitting requires matching registered chemistry. '
                    'Uncertain and provisional AC chemistry is excluded from shared fits.')
        elif mode == 'frozen':
            text = 'Apply saved parameters without fitting: ' + (str(self.frozen_path) if self.frozen_path else 'choose a native calibration.json first.')
        else:
            text = ('Apply the original DarSIA color-path, signal and flash models with the selected phase maps fixed. '
                    'Imported historical calibrations remain provisional until their physical and optical calibration is validated.')
        self.note.setText(text)

    def choose_frozen(self):
        from darsia.fluidflower.mass_calibration_library import series_root_for_results
        from darsia.fluidflower.tools.mass_calibration_library_selection import choose_saved_mass_calibration

        root = series_root_for_results((self.catalog or {}).get('results_root', ''))
        if root is not None:
            path = choose_saved_mass_calibration(self, root)
        else:
            path, _ = QFileDialog.getOpenFileName(self, 'Saved native calibration', '', 'JSON (*.json)')
        if path:
            self.frozen_path = Path(path)
            self.mode.setCurrentIndex(self.mode.findData('frozen')); self.update_note()

    def choose_overflow(self):
        row = self.table.currentRow()
        if not self.catalog or row < 0:
            self.note.setText('Select an experiment row first.'); return
        path, _ = QFileDialog.getOpenFileName(self, 'Reviewed overflow gas targets', '', 'JSON (*.json)')
        if path:
            run = self.catalog['jobs'][row]['run']
            self.overflow_paths[run] = path
            self.table.item(row, 4).setText(path)
            self.update_note()

    def clear_overflow(self):
        row = self.table.currentRow()
        if self.catalog and row >= 0:
            self.overflow_paths.pop(self.catalog['jobs'][row]['run'], None)
            self.table.item(row, 4).setText('None')
            self.update_note()

    def request(self, output):
        if not self.catalog:
            raise ValueError('Load a phase model first.')
        jobs = [job for row, job in enumerate(self.catalog['jobs'])
                if self.table.item(row, 0).checkState() == Qt.Checked]
        if not jobs:
            raise ValueError('Select at least one experiment with registered concentration inputs.')
        if any(not j.get('concentration', {}).get('ready') for j in jobs):
            raise ValueError('Selected concentration inputs are not ready.')
        mode = self.mode.currentData()
        if mode == 'series':
            if len(jobs) < 2:
                raise ValueError('Select at least two compatible experiments for a shared fit.')
            groups = {j['concentration'].get('chemistry_group') for j in jobs}
            if (len(groups) != 1 or groups & {'uncertainty', 'provisional', 'unconfirmed'}
                    or any(not j['concentration'].get('shared_fit_eligible', False) for j in jobs)):
                raise ValueError('A shared fit requires one confirmed chemistry group.')
        if mode == 'individual' and len(jobs) != 1:
            raise ValueError('Select exactly one experiment for an individual fit.')
        if mode == 'frozen' and not self.frozen_path:
            raise ValueError('Choose a saved native calibration first.')
        overflow_mode = self.overflow_mode.currentData()
        targets = {j['run']: self.overflow_paths[j['run']] for j in jobs if j['run'] in self.overflow_paths}
        if overflow_mode != 'off' and not targets:
            raise ValueError('Attach reviewed overflow targets or choose Without overflow.')
        if overflow_mode == 'constrain' and mode not in ('series', 'individual'):
            raise ValueError('Overflow constraints require an explicit fit; use comparison with frozen parameters.')
        return dict(format='native-studio-concentration-request-v1',
            model_root=self.catalog['model_root'], model_id=self.catalog['model_id'],
            runs=[j['run'] for j in jobs], mode=mode, overflow_mode=overflow_mode,
            overflow=targets if overflow_mode != 'off' else {},
            calibration_file=str(self.frozen_path) if mode == 'frozen' else None,
            output=str(output), results_root=self.catalog['results_root'])
