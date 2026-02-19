CREATE TABLE IF NOT EXISTS products_raw (
    id                      SERIAL PRIMARY KEY,
    productid               BIGINT NOT NULL,
    imageid                 BIGINT NOT NULL,
    prdtypecode             INTEGER NOT NULL,
    prodtype                VARCHAR(100) NOT NULL,
    product_designation     TEXT NOT NULL,
    product_description     TEXT,
    batch_id                VARCHAR(100) NOT NULL,
    dt_ingested             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(productid, batch_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_prdtypecode ON products_raw(prdtypecode);
CREATE INDEX IF NOT EXISTS idx_raw_batch ON products_raw(batch_id);

CREATE TABLE IF NOT EXISTS products_processed (
    id                      SERIAL PRIMARY KEY,
    productid               BIGINT NOT NULL,
    imageid                 BIGINT NOT NULL,
    prdtypecode             INTEGER NOT NULL,
    prodtype                VARCHAR(100) NOT NULL,
    designation             TEXT,
    description             TEXT,
    path_image_minio        VARCHAR(500),
    image_exists            BOOLEAN DEFAULT FALSE,
    batch_id                VARCHAR(100) NOT NULL,
    dt_processed            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(productid, batch_id)
);

CREATE INDEX IF NOT EXISTS idx_processed_prdtypecode ON products_processed(prdtypecode);
CREATE INDEX IF NOT EXISTS idx_processed_batch ON products_processed(batch_id);

-- Latest version of each product (most recent batch wins)
CREATE OR REPLACE VIEW products_latest AS
SELECT DISTINCT ON (productid) *
FROM products_processed
ORDER BY productid, dt_processed DESC;
