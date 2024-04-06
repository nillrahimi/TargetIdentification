import os.path
import pandas as pd


class Converter:
    def __init__(self, from_file_path: str = "", to_file_path: str = ""):
        self.from_file_path = from_file_path
        self.to_file_path = to_file_path

    def convertXlsxToCsv(
        self, edges_filename: str = "edges", feature_filename: str = "nodes"
    ) -> None:
        """

        :param edges_filename:
        :param feature_filename:
        :return:
        """
        edges_filename_xlsx = os.path.join(
            self.from_file_path, edges_filename + ".xlsx"
        )
        nodes_filename_xlsx = os.path.join(
            self.from_file_path, feature_filename + ".xlsx"
        )
        edges_filename_csv = os.path.join(self.to_file_path, edges_filename + ".csv")
        nodes_filename_csv = os.path.join(self.to_file_path, feature_filename + ".csv")
        # Read edges.xlsx file and convert to edges.csv
        edges_df = pd.read_excel(edges_filename_xlsx)
        edges_df = edges_df.drop(edges_df.filter(regex="Unnamed").columns, axis=1)
        edges_df.to_csv(edges_filename_csv, index=False, header=["id_1", "id_2"])

        # Read nodes.xlsx file and convert to feature.csv
        nodes_df = pd.read_excel(nodes_filename_xlsx)
        nodes_df = nodes_df.drop(nodes_df.filter(regex="Unnamed").columns, axis=1)
        nodes_df.to_csv(nodes_filename_csv, index=False, header=["node_id", "labels"])
