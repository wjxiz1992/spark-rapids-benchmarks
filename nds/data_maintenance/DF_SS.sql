--
-- SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
-- http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
--
-- -----
--
-- Certain portions of the contents of this file are derived from TPC-DS version 3.2.0
-- (retrieved from www.tpc.org/tpc_documents_current_versions/current_specifications5.asp).
-- Such portions are subject to copyrights held by Transaction Processing Performance Council (“TPC”)
-- and licensed under the TPC EULA (a copy of which accompanies this file as “TPC EULA” and is also
-- available at http://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp) (the “TPC EULA”).
--
-- You may not use this file except in compliance with the TPC EULA.
-- DISCLAIMER: Portions of this file is derived from the TPC-DS Benchmark and as such any results
-- obtained using this file are not comparable to published TPC-DS Benchmark results, as the results
-- obtained from using this file do not comply with the TPC-DS Benchmark.
--

delete from store_returns where sr_ticket_number in (select ss_ticket_number from store_sales, date_dim   where ss_sold_date_sk=d_date_sk and d_date between 'DATE1' and 'DATE2');
delete from store_sales where ss_sold_date_sk >= (select min(d_date_sk) from date_dim  where d_date between 'DATE1' and 'DATE2') and
                ss_sold_date_sk <= (select max(d_date_sk) from date_dim  where d_date between 'DATE1' and 'DATE2');
