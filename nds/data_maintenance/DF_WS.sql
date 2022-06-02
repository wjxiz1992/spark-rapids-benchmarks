select ws_order_number from web_sales, date_dim where ws_sold_date_sk=d_date_sk and d_date between 'DATE1' and 'DATE2';
select min(d_date_sk) from date_dim where d_date between 'DATE1' and 'DATE2';
select max(d_date_sk) from date_dim where d_date between 'DATE1' and 'DATE2';
delete from web_returns where wr_order_number in (SQL1);
delete from web_sales where ws_sold_date_sk >= (SQL2) and
                 ws_sold_date_sk <= (SQL3);
