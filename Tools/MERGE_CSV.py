import sys
import csv
import bisect
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QGridLayout,
    QMessageBox,
)


class CsvMerger(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # 输入文件控件
        self.re_mode_label = QLabel("re(Mode) CSV:")
        self.re_mode_input = QLineEdit()
        self.re_mode_btn = QPushButton("Browse...", self)

        self.vpc_label = QLabel("vpc CSV:")
        self.vpc_input = QLineEdit()
        self.vpc_btn = QPushButton("Browse...", self)

        self.zk_abs_label = QLabel("Zk_Abs CSV:")
        self.zk_abs_input = QLineEdit()
        self.zk_abs_btn = QPushButton("Browse...", self)

        # 输出文件控件
        self.output_label = QLabel("Output CSV:")
        self.output_input = QLineEdit()
        self.output_btn = QPushButton("Browse...", self)

        # 操作按钮
        self.merge_btn = QPushButton("Merge Files", self)
        self.interpolate_btn = QPushButton("Merge with Interpolation", self)

        # 布局设置
        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(self.re_mode_label, 0, 0)
        grid.addWidget(self.re_mode_input, 0, 1)
        grid.addWidget(self.re_mode_btn, 0, 2)

        grid.addWidget(self.vpc_label, 1, 0)
        grid.addWidget(self.vpc_input, 1, 1)
        grid.addWidget(self.vpc_btn, 1, 2)

        grid.addWidget(self.zk_abs_label, 2, 0)
        grid.addWidget(self.zk_abs_input, 2, 1)
        grid.addWidget(self.zk_abs_btn, 2, 2)

        grid.addWidget(self.output_label, 3, 0)
        grid.addWidget(self.output_input, 3, 1)
        grid.addWidget(self.output_btn, 3, 2)

        grid.addWidget(self.merge_btn, 4, 1)
        grid.addWidget(self.interpolate_btn, 4, 2)

        self.setLayout(grid)

        # 信号连接
        self.re_mode_btn.clicked.connect(lambda: self.browse_file(self.re_mode_input))
        self.vpc_btn.clicked.connect(lambda: self.browse_file(self.vpc_input))
        self.zk_abs_btn.clicked.connect(lambda: self.browse_file(self.zk_abs_input))
        self.output_btn.clicked.connect(self.browse_output)
        self.merge_btn.clicked.connect(self.merge_files)
        self.interpolate_btn.clicked.connect(self.merge_with_interpolation)

        # 窗口设置
        self.setGeometry(300, 300, 750, 220)
        self.setWindowTitle("CSV Data Merger with Interpolation")
        self.show()

    def browse_file(self, target_input):
        """选择输入文件"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)"
        )
        if filename:
            target_input.setText(filename)

    def browse_output(self):
        """选择输出文件"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Output File", "", "CSV Files (*.csv)"
        )
        if filename:
            self.output_input.setText(filename)

    def load_csv_data(self, filename, phi_col, value_col):
        """读取CSV数据，指定列索引"""
        data = {}
        try:
            with open(filename, "r", encoding="utf-8-sig") as file:
                reader = csv.reader(file)
                next(reader)  # 跳过首行

                for row_num, row in enumerate(reader, start=2):
                    if len(row) < max(phi_col, value_col) + 1:
                        continue

                    try:
                        # 数据清洗
                        phi_str = row[phi_col].strip().replace(",", ".")
                        value_str = (
                            row[value_col].strip().replace(",", "").replace(" ", "")
                        )

                        phi = round(float(phi_str), 2)
                        value = float(value_str)

                        if phi in data:
                            QMessageBox.warning(
                                self,
                                "Duplicate Data",
                                f"重复的phi值 {phi} 在文件 {filename} 第{row_num}行",
                            )
                            continue

                        data[phi] = value
                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            "数据错误",
                            f"{filename} 第{row_num}行格式无效\n{str(e)}",
                        )
                        continue
            return data
        except Exception as e:
            QMessageBox.critical(self, "文件错误", f"无法读取 {filename}\n{str(e)}")
            return None

    def merge_files(self):
        """执行合并操作"""
        input_files = [
            self.re_mode_input.text(),
            self.vpc_input.text(),
            self.zk_abs_input.text(),
        ]
        output_file = self.output_input.text()

        if not all(input_files) or not output_file:
            QMessageBox.warning(
                self, "Warning", "Please select all input and output files"
            )
            return

        try:
            # 加载数据（指定不同文件的列索引）
            re_data = self.load_csv_data(input_files[0], phi_col=0, value_col=1)
            vpc_data = self.load_csv_data(input_files[1], phi_col=0, value_col=1)
            zk_data = self.load_csv_data(input_files[2], phi_col=1, value_col=2)

            if None in (re_data, vpc_data, zk_data):
                return

            # 获取共同phi值
            common_phi = (
                set(re_data.keys()) & set(vpc_data.keys()) & set(zk_data.keys())
            )
            if not common_phi:
                raise ValueError("No common phi values found in all files")

            # 排序phi值
            sorted_phi = sorted(common_phi)

            # 写入输出文件
            with open(output_file, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["#Zk_Abs", "0", "re(Mode(1))", "vpc"])

                success_count = 0
                for phi in sorted_phi:
                    try:
                        writer.writerow(
                            [
                                zk_data[phi],
                                0,
                                re_data[phi] / 1e9,  # 单位转换
                                vpc_data[phi],
                            ]
                        )
                        success_count += 1
                    except KeyError:
                        continue

                if success_count == 0:
                    raise ValueError("No valid data merged")

            QMessageBox.information(
                self,
                "Success",
                f"Merged {success_count} data points\n"
                f"Output saved to:\n{output_file}",
            )

        except PermissionError:
            QMessageBox.critical(self, "Error", "No permission to write output file")
        except Exception as e:
            QMessageBox.critical(self, "Merge Error", f"Merge failed: {str(e)}")

    def merge_with_interpolation(self):
        """带插值的合并操作"""
        input_files = [
            self.re_mode_input.text(),
            self.vpc_input.text(),
            self.zk_abs_input.text(),
        ]
        output_file = self.output_input.text()

        if not all(input_files) or not output_file:
            QMessageBox.warning(self, "Warning", "请选择所有输入和输出文件")
            return

        try:
            # 加载数据（指定不同文件的列索引）
            re_data = self.load_csv_data(input_files[0], phi_col=0, value_col=1)
            vpc_data = self.load_csv_data(input_files[1], phi_col=0, value_col=1)
            zk_data = self.load_csv_data(input_files[2], phi_col=1, value_col=2)

            if None in (re_data, vpc_data, zk_data):
                return

            # 合并基础数据
            common_phi = (
                set(re_data.keys()) & set(vpc_data.keys()) & set(zk_data.keys())
            )
            if not common_phi:
                raise ValueError("所有文件中没有共同的phi值")

            sorted_phi = sorted(common_phi)
            merged_data = []
            for phi in sorted_phi:
                try:
                    merged_data.append(
                        [zk_data[phi], 0, re_data[phi] / 1e9, vpc_data[phi]]
                    )
                except KeyError:
                    continue

            if not merged_data:
                raise ValueError("没有有效数据可供合并")

            # 插值处理
            merged_data.sort(key=lambda x: x[2])
            re_values = [row[2] for row in merged_data]

            min_re = int(min(re_values))
            max_re = int(max(re_values)) + 1

            interpolated = []
            for target_re in range(min_re, max_re + 1):
                index = bisect.bisect_left(re_values, target_re)

                # 边界处理
                if index == 0 or index >= len(re_values):
                    continue

                # 精确匹配处理
                if re_values[index - 1] == target_re:
                    interpolated.append(merged_data[index - 1])
                    continue
                if index < len(re_values) and re_values[index] == target_re:
                    interpolated.append(merged_data[index])
                    continue

                # 计算插值
                prev_re = re_values[index - 1]
                next_re = re_values[index]
                if not (prev_re < target_re < next_re):
                    continue

                alpha = (target_re - prev_re) / (next_re - prev_re)
                prev_row = merged_data[index - 1]
                next_row = merged_data[index]

                zk_interp = prev_row[0] + alpha * (next_row[0] - prev_row[0])
                vpc_interp = prev_row[3] + alpha * (next_row[3] - prev_row[3])

                interpolated.append([zk_interp, 0, target_re, vpc_interp])

            # 保存结果
            with open(output_file, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["#Zk_Abs", "0", "re(Mode(1))", "vpc"])
                writer.writerows(interpolated)

            QMessageBox.information(
                self,
                "Success",
                f"生成 {len(interpolated)} 个插值点\n"
                f"输出文件已保存至:\n{output_file}",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"插值失败: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = CsvMerger()
    sys.exit(app.exec_())
