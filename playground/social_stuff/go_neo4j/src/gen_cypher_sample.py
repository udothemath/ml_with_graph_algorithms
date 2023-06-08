import logging
import setting


class gen_csv_file:
    def __init__(self, id_prefix='ab', num_rows=10, path=NEO4j_DIR, logger=None):
        """
            generate sample csv file
        """
        self.__id_prefix = id_prefix
        self.__num_rows = num_rows
        self.__path = path

        if logger is None:
            logging.basicConfig(
                filename=f'{NEO4j_DIR}/logs1.log',
                filemode='a',
                format='%(asctime)s: %(name)s %(levelname)s %(message)s',
                datefmt='%H:%M:%S',
                level=logging.INFO
            )

            logger = logging.getLogger(__name__)
            logger.info(
                "===============================================================================")
            logger.info("Starting Application Logging")

            # logger.info( "Logging Config Imported in Second Script" )

        self.logger = logger

    def gen_node(self, num_cols=2) -> None:
        """
        generate csv file for node

        Parameters:
        num_rows (int): number of rows
        num_cols (int): number of columns

        Returns:
        None. The csv file will be generated at desired path

        Raises:
        TypeError: If num_rows or num_cols is not an integer
        """
        if not isinstance(self.__num_rows, int) or not isinstance(num_cols, int):
            raise TypeError("Input must be a number.")

        file_with_path = os.path.join(
            self.__path, f"data_node_r{self.__num_rows}_c{num_cols}.csv")
        col_name = ['id', 'value']
        id_list = [f"{self.__id_prefix}_{i}" for i in range(self.__num_rows)]
        value_list = [randrange(0, 200) for i in range(self.__num_rows)]
        rows = zip(id_list, value_list)

        # Open a new CSV file for writing
        with open(file_with_path, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)

            # Write the header row
            csvwriter.writerow(col_name)

            # for word in the_list:
            for row in rows:
                csvwriter.writerow(row)

        print(
            f"Ur file with rows:{self.__num_rows} and cols:{num_cols} is ready.")

    @wrap_log
    def gen_relation(self, num_cols=2) -> None:
        """
        generate csv file for relation

        Parameters:
        num_rows (int): number of rows
        num_cols (int): number of columns

        Returns:
        None. The csv file will be generated at desired path

        Raises:
        TypeError: If num_rows or num_cols is not an integer
        """
        if not isinstance(self.__num_rows, int) or not isinstance(num_cols, int):
            raise TypeError("Input must be a number.")

        file_with_path = os.path.join(self.__path,
                                      f"data_relation_r{self.__num_rows}_c{num_cols}.csv")
        col_name = ['from', 'to']
        half_index = int(self.__num_rows/2)
        from_id = [f"{self.__id_prefix}_{2*i}" for i in range(half_index)]
        to_id = [f"{self.__id_prefix}_{2*i+1}" for i in range(half_index)]
        rows = zip(from_id, to_id)

        # Open a new CSV file for writing
        with open(file_with_path, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)

            # Write the header row
            csvwriter.writerow(col_name)

            # for word in the_list:
            for row in rows:
                csvwriter.writerow(row)

        self.logger.info(
            f"Ur file of relation with rows:{self.__num_rows} is ready.")

    @wrap_log
    def gen_relation_random(self, num_cols=2, repeated_rel=1) -> None:
        """
        generate csv file for relation

        Parameters:
        num_rows (int): number of rows
        num_cols (int): number of columns

        Returns:
        None. The csv file will be generated at desired path

        Raises:
        TypeError: If num_rows or num_cols is not an integer
        """
        if not isinstance(self.__num_rows, int) or not isinstance(num_cols, int):
            raise TypeError("Input must be a number.")

        file_with_path = os.path.join(self.__path,
                                      f"data_relation_r{self.__num_rows}_c{num_cols}.csv")
        col_name = ['from', 'to']
        from_id_all = []
        to_id_all = []
        for _ in range(repeated_rel):
            from_id_all.extend(
                f"{self.__id_prefix}_{i}" for i in range(self.__num_rows))
            to_id_all.extend(
                f"{self.__id_prefix}_{randrange(self.__num_rows-1)}" for i in range(self.__num_rows))
        rows = zip(from_id_all, to_id_all)
        # Open a new CSV file for writing
        with open(file_with_path, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)

            # Write the header row
            csvwriter.writerow(col_name)

            # for word in the_list:
            for row in rows:
                csvwriter.writerow(row)

        self.logger.info(
            f"Ur file of relation with rows:{self.__num_rows} is ready.")

    @wrap_log
    def gen_relation_random_with_prop(self, num_cols=2, repeated_rel=1) -> None:
        """
        generate csv file for relation

        Parameters:
        num_rows (int): number of rows
        num_cols (int): number of columns

        Returns:
        None. The csv file will be generated at desired path

        Raises:
        TypeError: If num_rows or num_cols is not an integer
        """
        if not isinstance(self.__num_rows, int) or not isinstance(num_cols, int):
            raise TypeError("Input must be a number.")

        file_with_path = os.path.join(self.__path,
                                      f"data_relation_r{self.__num_rows}_c{num_cols}.csv")
        col_name = ['from', 'to', 'from_value', 'to_value']
        from_id_all = []
        to_id_all = []
        from_value_all = []
        to_value_all = []
        for _ in range(repeated_rel):
            from_id_all.extend(
                f"{self.__id_prefix}_{i}" for i in range(self.__num_rows))
            to_id_all.extend(
                f"{self.__id_prefix}_{randrange(self.__num_rows-1)}" for i in range(self.__num_rows))
            from_value_all.extend(randrange(0, 200)
                                  for i in range(self.__num_rows))
            to_value_all.extend(randrange(0, 200)
                                for i in range(self.__num_rows))
        rows = zip(from_id_all, to_id_all, from_value_all, to_value_all)
        # Open a new CSV file for writing
        with open(file_with_path, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)

            # Write the header row
            csvwriter.writerow(col_name)

            # for word in the_list:
            for row in rows:
                csvwriter.writerow(row)

        self.logger.info(
            f"Ur file of relation with rows:{self.__num_rows} is ready.")
