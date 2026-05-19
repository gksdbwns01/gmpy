from ..utils import *
from ..core import *

class IntegrityThread(QThread):
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, img_dir, lbl_dir, num_classes):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.num_classes = num_classes

    def run(self):
        issues = []
        try:
            img_files = iter_supported_images(self.img_dir)
            lbl_files = [f for f in self.lbl_dir.iterdir() if f.suffix.lower() == ".txt"]

            img_stems = {f.stem: f for f in img_files}
            lbl_stems = {f.stem: f for f in lbl_files}

            all_stems = set(img_stems.keys()).union(set(lbl_stems.keys()))
            total = len(all_stems)
            hashes = {}

            for i, stem in enumerate(all_stems):
                if total > 0:
                    self.progress.emit(int((i / total) * 100), f"무결성 검사 중... ({i}/{total})")

                # 1. 미스매치 검사
                if stem not in img_stems:
                    issues.append({"file": lbl_stems[stem].name, "type": "이미지 누락", "desc": "라벨은 있지만 짝이 되는 이미지 파일이 없습니다."})
                    continue
                if stem not in lbl_stems:
                    issues.append({"file": img_stems[stem].name, "type": "라벨 누락", "desc": "이미지는 있지만 짝이 되는 라벨(.txt) 파일이 없습니다."})
                    continue

                img_path = img_stems[stem]
                lbl_path = lbl_stems[stem]

                # 2. 이미지 손상 및 중복(Hash) 검사
                img = cv2.imread(str(img_path))
                if img is None:
                    issues.append({"file": img_path.name, "type": "이미지 손상", "desc": "이미지 파일을 읽을 수 없거나 깨졌습니다."})
                    continue
                else:
                    # 초고속 dHash 생성 (8x8)
                    resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8))
                    diff = resized[:, 1:] > resized[:, :-1]
                    dhash = sum([2 ** idx for (idx, v) in enumerate(diff.flatten()) if v])
                    
                    if dhash in hashes:
                        issues.append({"file": img_path.name, "type": "이미지 중복", "desc": f"'{hashes[dhash]}' 파일과 이미지가 완벽히 동일합니다. (Data Leakage 위험)"})
                    else:
                        hashes[dhash] = img_path.name

                # 3. 빈 라벨 검사
                if lbl_path.stat().st_size == 0:
                    issues.append({"file": lbl_path.name, "type": "빈 라벨", "desc": "라벨 파일이 0 바이트로 비어있습니다. (객체 없음)"})
                    continue

                lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
                if not lines:
                    issues.append({"file": lbl_path.name, "type": "빈 라벨", "desc": "라벨 파일 내부에 내용이 없습니다."})
                    continue

                # 4. Class ID 및 BBox 유효성 검사
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        issues.append({"file": lbl_path.name, "type": "포맷 오류", "desc": f"{line_idx+1}번째 줄: YOLO 포맷(값 5개)이 아닙니다."})
                        continue

                    try:
                        cls_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])

                        if cls_id < 0 or cls_id >= self.num_classes:
                            issues.append({"file": lbl_path.name, "type": "Class ID 범위 오류", "desc": f"{line_idx+1}번째 줄: ID({cls_id})가 설정된 클래스 범위를 벗어납니다."})

                        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                            issues.append({"file": lbl_path.name, "type": "범위 이탈 (Out-of-bounds)", "desc": f"{line_idx+1}번째 줄: 중심 좌표(x,y)가 0~1 범위를 벗어납니다."})

                        if w <= 0.001 or h <= 0.001:
                            issues.append({"file": lbl_path.name, "type": "극단적 BBox (Extreme)", "desc": f"{line_idx+1}번째 줄: BBox 너비나 높이가 너무 작습니다 (0.001 이하)."})
                        elif w > 1.05 or h > 1.05:
                            issues.append({"file": lbl_path.name, "type": "범위 이탈 (Out-of-bounds)", "desc": f"{line_idx+1}번째 줄: BBox 크기가 이미지 크기를 과도하게 벗어납니다."})

                    except ValueError:
                        issues.append({"file": lbl_path.name, "type": "값 오류", "desc": f"{line_idx+1}번째 줄: 문자가 섞여 있거나 숫자로 변환할 수 없습니다."})

            self.progress.emit(100, "검사 완료")
            self.finished_ok.emit(issues)
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())
