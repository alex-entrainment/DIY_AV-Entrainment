use once_cell::sync::Lazy;

const TABLE_SIZE: usize = 1 << 16; // 65536 steps
const TWO_PI: f32 = std::f32::consts::PI * 2.0;

pub struct TrigLut {
    table: Vec<f32>,
}

impl TrigLut {
    fn new() -> Self {
        let mut table = Vec::with_capacity(TABLE_SIZE + 1);
        for i in 0..=TABLE_SIZE {
            let angle = i as f32 / TABLE_SIZE as f32 * TWO_PI;
            table.push(angle.sin());
        }
        Self { table }
    }

    #[inline(always)]
    fn sin(&self, angle: f32) -> f32 {
        let pos = angle.rem_euclid(TWO_PI) / TWO_PI * TABLE_SIZE as f32;
        let idx = pos.floor() as usize;
        let frac = pos - idx as f32;
        let a = self.table[idx];
        let b = self.table[idx + 1];
        a + (b - a) * frac
    }

    #[inline(always)]
    fn cos(&self, angle: f32) -> f32 {
        self.sin(angle + std::f32::consts::FRAC_PI_2)
    }
}

pub static LUT: Lazy<TrigLut> = Lazy::new(TrigLut::new);

#[inline(always)]
pub fn sin_lut(angle: f32) -> f32 {
    LUT.sin(angle)
}

#[inline(always)]
pub fn cos_lut(angle: f32) -> f32 {
    LUT.cos(angle)
}
