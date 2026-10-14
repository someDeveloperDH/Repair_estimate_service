"""파손 분류 서버(serve.py) 시험 GUI. 표준 라이브러리만 사용.
pet_eye 실습 GUI를 바탕으로 응답 형식(part / confidence / probs / threshold)에 맞게 고친 것."""
import csv
import json
import mimetypes
import queue
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
PARTS = ['outsole', 'heel']          # 클래스마다 독립 확률. 둘 다 threshold 미만이면 other
PART_KO = {'outsole': '아웃솔', 'heel': '뒤꿈치', 'other': '그 외'}
FIELDS = ['file', 'path', 'status', 'part', 'part_ko', 'confidence', 'p_outsole', 'p_heel', 'threshold',
          'inference_ms', 'seconds', 'attempt', 'response_format', 'response', 'raw_json', 'validation_note', 'error']


def request_api(base, endpoint, params=None, path=None, timeout=120):
    base = base.strip().rstrip('/')
    parts = urlsplit(base)
    if parts.scheme not in ('http', 'https') or not parts.netloc or parts.query or parts.fragment:
        raise ValueError('http(s)로 시작하는 서버 기본 주소를 입력하세요.')
    headers = {'Accept': 'application/json', 'ngrok-skip-browser-warning': 'true'}
    body = None
    if path is not None:
        path = Path(path)
        boundary = 'shoe' + uuid.uuid4().hex
        name = 'image' + path.suffix.lower()
        mime = mimetypes.guess_type(name)[0] or 'application/octet-stream'
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: {mime}\r\n\r\n').encode() + path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
        headers['Content-Type'] = 'multipart/form-data; boundary=' + boundary
    url = base + endpoint + ('?' + urlencode(params) if params else '')
    try:
        with urlopen(Request(url, data=body, headers=headers), timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        raise ValueError(f'HTTP {exc.code}: {exc.read().decode("utf-8", errors="replace")[:500]}') from exc
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('JSON 응답이 아닙니다. 서버 주소와 ngrok 상태를 확인하세요.') from exc


def validate(data):
    """서버 응답 형식 검증 후 표시용 1행으로 변환."""
    if not isinstance(data, dict) or any(k not in data for k in ('part', 'confidence', 'probs', 'threshold')):
        raise ValueError('part/confidence/probs/threshold 필드 누락')
    probs, thr = data['probs'], data['threshold']
    if not isinstance(probs, dict) or not probs:
        raise ValueError('probs가 비어 있습니다.')
    for key, value in list(probs.items()) + [('confidence', data['confidence']), ('threshold', thr)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError('확률 범위 오류: ' + key)
    top = max(probs.values())
    # 서버가 소수점 4자리로 반올림하므로 경계 오차를 허용합니다.
    if abs(top - thr) > 0.000051:
        if data['part'] == 'other':
            if top >= thr:
                raise ValueError('기준 이상인 부위가 있는데 그 외로 판정')
        elif data['part'] not in probs or abs(probs[data['part']] - top) > 0.000051 or top < thr:
            raise ValueError('판정 부위와 확률 불일치')
    expect = probs.get(data['part'], 0) if data['part'] != 'other' else 1 - top
    if abs(data['confidence'] - expect) > 0.00011:
        raise ValueError('확신도 계산 불일치')
    return {'part': data['part'], 'part_ko': data.get('part_ko') or PART_KO.get(data['part'], data['part']),
            'confidence': data['confidence'], 'threshold': thr, 'inference_ms': data.get('inference_ms', ''),
            **{'p_' + p: probs.get(p, '') for p in PARTS}}


def display_response(data):
    """모든 JSON을 보존하고 알려진 형식만 전용 열에 표시합니다."""
    raw = json.dumps(data, ensure_ascii=False, allow_nan=False)
    summary = raw if len(raw) <= 300 else raw[:300] + '…'
    try:
        row = validate(data)
        return [dict(row, response_format='shoe-part', response=summary, raw_json=raw, validation_note='')]
    except ValueError as exc:
        return [dict(response_format='JSON', response=summary, raw_json=raw, validation_note=str(exc))]


def save_csv(path, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@', '\t', '\r')) else v for k, v in row.items()})


def run_batch(base, files, stop, emit):
    """1차 처리 후 실패 파일만 최대 5회 재시도. 중지 시 재시도하지 않습니다."""
    pending = []
    outcomes = {}
    completed = 0
    for attempt in range(1, 7):
        targets = files if attempt == 1 else pending
        if not targets or stop.is_set():
            break
        if attempt > 1:
            emit('status', f'재시도 {attempt-1}/5 · 실패 이미지 {len(targets)}개 처리 중...')
            emit('progress_setup', completed + len(targets))
        pending = []
        for index, path in enumerate(targets, 1):
            if stop.is_set():
                break
            start = time.perf_counter()
            common = dict(file=path.name, path=str(path), attempt=attempt)
            try:
                result = display_response(request_api(base, '/predict', path=path))
                elapsed = round(time.perf_counter() - start, 3)
                rows = [{**common, **row, 'status': '성공', 'seconds': elapsed, 'error': ''} for row in result]
                outcomes[str(path)] = True
            except Exception as exc:
                rows = [dict(common, status='실패', seconds=round(time.perf_counter() - start, 3), error=str(exc))]
                outcomes[str(path)] = False
                pending.append(path)
            emit('result', (str(path), rows))
            completed += 1
            emit('progress', completed)
            success = sum(outcomes.values())
            emit('status', f"{'1차 처리' if attempt == 1 else f'재시도 {attempt-1}/5'} {index}/{len(targets)} · 이미지 성공 {success} · 실패 {len(outcomes)-success}")
    success = sum(outcomes.values())
    emit('status', f"{'중지됨' if stop.is_set() else '완료'} · 이미지 성공 {success} · 실패 {len(outcomes)-success} · 미처리 {len(files)-len(outcomes)} · CSV 저장 가능")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Mendii Damage Classifier')
        self.geometry('1180x720')
        self.files, self.rows, self.buttons = [], [], []
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.busy = False
        self.url = tk.StringVar(value='http://127.0.0.1:8000')
        self.status = tk.StringVar(value='서버 주소를 확인하고 서버 점검을 누르세요.')
        frame = ttk.Frame(self, padding=12); frame.pack(fill='both', expand=True)
        bar = ttk.Frame(frame); bar.pack(fill='x')
        ttk.Label(bar, text='API 주소').pack(side='left')
        ttk.Entry(bar, textvariable=self.url).pack(side='left', fill='x', expand=True, padx=8)
        self.button(bar, '서버 점검', self.health)
        bar = ttk.Frame(frame); bar.pack(fill='x')
        for label, func in [('여러 파일 추가', self.add_files), ('폴더 추가', self.add_folder), ('목록 비우기', self.clear), ('추론 시작', self.predict), ('CSV 저장', self.export)]:
            self.button(bar, label, func)
        ttk.Button(bar, text='중지', command=self.stop.set).pack(side='left', padx=4)
        self.count = tk.StringVar(value='선택 이미지 0개')
        ttk.Label(frame, textvariable=self.count).pack(anchor='w', pady=5)
        self.listbox = tk.Listbox(frame, height=5); self.listbox.pack(fill='x')
        area = ttk.Frame(frame); area.pack(fill='both', expand=True, pady=8)
        columns = ('file', 'status', 'part_ko', 'confidence', 'p_outsole', 'p_heel', 'threshold', 'inference_ms', 'seconds', 'response', 'error')
        self.table = ttk.Treeview(area, columns=columns, show='headings')
        for key, label in zip(columns, ['파일', '상태', '판정', '확신도', '아웃솔', '뒤꿈치', '기준값', '모델(ms)', '응답(초)', 'JSON 응답', '오류']):
            self.table.heading(key, text=label); self.table.column(key, width=160 if key in ('file', 'response', 'error') else 85)
        ys = ttk.Scrollbar(area, orient='vertical', command=self.table.yview)
        xs = ttk.Scrollbar(area, orient='horizontal', command=self.table.xview)
        self.table.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.table.grid(row=0, column=0, sticky='nsew'); ys.grid(row=0, column=1, sticky='ns'); xs.grid(row=1, column=0, sticky='ew')
        area.rowconfigure(0, weight=1); area.columnconfigure(0, weight=1)
        self.table.bind('<<TreeviewSelect>>', self.details)
        self.detail = tk.StringVar(value='행을 선택하면 원본 경로와 오류가 표시됩니다.')
        ttk.Label(frame, textvariable=self.detail, wraplength=1100).pack(anchor='w')
        json_area = ttk.Frame(frame); json_area.pack(fill='x', pady=5)
        self.json_view = tk.Text(json_area, height=8, wrap='word', state='disabled')
        json_scroll = ttk.Scrollbar(json_area, orient='vertical', command=self.json_view.yview)
        self.json_view.configure(yscrollcommand=json_scroll.set)
        self.json_view.pack(side='left', fill='x', expand=True)
        json_scroll.pack(side='right', fill='y')
        self.progress = ttk.Progressbar(frame); self.progress.pack(fill='x', pady=5)
        ttk.Label(frame, textvariable=self.status, wraplength=1100).pack(anchor='w')
        ttk.Label(frame, text='모델 출력은 참고용입니다. 서버 응답 성공이 분류 정확도를 보장하지 않습니다.').pack(anchor='w', pady=5)
        self.after(100, self.poll)

    def button(self, parent, label, func):
        button = ttk.Button(parent, text=label, command=func)
        button.pack(side='left', padx=4); self.buttons.append(button)

    def add(self, paths):
        seen = set(self.files)
        for item in paths:
            path = Path(item).resolve()
            if path.is_file() and path.suffix.lower() in EXTENSIONS and path not in seen:
                self.files.append(path); seen.add(path); self.listbox.insert('end', str(path))
        self.count.set(f'선택 이미지 {len(self.files)}개')

    def add_files(self):
        self.add(filedialog.askopenfilenames(filetypes=[('이미지', '*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff'), ('모든 파일', '*.*')]))

    def add_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.add(sorted(Path(folder).rglob('*')))

    def clear(self):
        self.files.clear(); self.rows.clear(); self.listbox.delete(0, 'end')
        self.table.delete(*self.table.get_children()); self.count.set('선택 이미지 0개')
        self.json_view.configure(state='normal'); self.json_view.delete('1.0', 'end'); self.json_view.configure(state='disabled')

    def launch(self, worker):
        if self.busy: return
        self.busy = True; self.stop.clear()
        for button in self.buttons: button.configure(state='disabled')
        threading.Thread(target=worker, daemon=True).start()

    def health(self):
        base = self.url.get(); self.status.set('서버 점검 중...')
        def worker():
            try:
                start = time.perf_counter(); data = request_api(base, '/health', timeout=15)
                summary = display_response(data)[0]['response']
                self.events.put(('status', f'서버 JSON 응답 수신 · {time.perf_counter()-start:.2f}초 · {summary}'))
            except Exception as exc: self.events.put(('status', f'점검 실패: {exc}'))
            finally: self.events.put(('done', None))
        self.launch(worker)

    def predict(self):
        if not self.files:
            messagebox.showinfo('이미지 선택', '파일 또는 폴더를 먼저 추가하세요.'); return
        base, files = self.url.get(), list(self.files)
        self.rows.clear(); self.table.delete(*self.table.get_children())
        self.progress.configure(maximum=len(files), value=0); self.status.set('전송 중...')
        def worker():
            try:
                run_batch(base, files, self.stop, lambda kind, value: self.events.put((kind, value)))
            except Exception as exc:
                self.events.put(('status', f'처리 오류: {exc}'))
            finally:
                self.events.put(('done', None))
        self.launch(worker)

    def poll(self):
        for _ in range(200):
            try: kind, value = self.events.get_nowait()
            except queue.Empty: break
            if kind == 'result':
                file_path, results = value
                # 재시도 결과는 기존 실패 행을 대체하여 CSV에도 최종 결과만 남깁니다.
                self.rows[:] = [row for row in self.rows if row['path'] != file_path]
                self.rows.extend(results)
                self.table.delete(*self.table.get_children())
                pct = ('confidence', 'p_outsole', 'p_heel')
                for index, row in enumerate(self.rows):
                    values = [f'{row[k]:.1%}' if k in pct and isinstance(row.get(k), (int, float)) else row.get(k, '') for k in self.table['columns']]
                    self.table.insert('', 'end', iid=str(index), values=values)
                self.detail.set('행을 선택하면 원본 경로와 전체 JSON이 표시됩니다.')
            elif kind == 'progress_setup':
                self.progress.configure(maximum=value)
            elif kind == 'status': self.status.set(value)
            elif kind == 'progress': self.progress.configure(value=value)
            elif kind == 'done':
                self.busy = False
                for button in self.buttons: button.configure(state='normal')
        self.after(100, self.poll)

    def details(self, event=None):
        selected = self.table.selection()
        if selected:
            row = self.rows[int(selected[0])]
            self.detail.set(row['path'] + (' · ' + row['error'] if row.get('error') else '') + (' · 형식 검증: ' + row['validation_note'] if row.get('validation_note') else ''))
            raw = row.get('raw_json', '')
            rendered = json.dumps(json.loads(raw), ensure_ascii=False, indent=2) if raw else row.get('error', '')
            self.json_view.configure(state='normal')
            self.json_view.delete('1.0', 'end')
            self.json_view.insert('1.0', rendered)
            self.json_view.configure(state='disabled')

    def export(self):
        if not self.rows:
            messagebox.showinfo('CSV 저장', '결과가 없습니다.'); return
        path = filedialog.asksaveasfilename(defaultextension='.csv', initialfile='damage_cls_results.csv', filetypes=[('CSV', '*.csv')])
        if path:
            try:
                save_csv(path, self.rows); messagebox.showinfo('저장 완료', f'{len(self.rows)}행 저장\n{path}')
            except OSError as exc: messagebox.showerror('저장 실패', str(exc))


if __name__ == '__main__':
    App().mainloop()
