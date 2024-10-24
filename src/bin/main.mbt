let emit_js : Ref[Bool] = Ref::new(false)

let emit_wasm : Ref[Bool] = Ref::new(false)

/// The stage to be run next.
enum Stages {
  Parse
  Typecheck
  PreCps
  Cps
  CloPS
  // NOTE: add stages here.
  Js
  SsaCfg
  Asm
  Finished
} derive(Show, Eq, Compare)

fn Stages::from_string(s : String) -> Stages? {
  match s {
    "parse" => Some(Stages::Parse)
    "typecheck" => Some(Stages::Typecheck)
    "precps" => Some(Stages::PreCps)
    "cps" => Some(Stages::Cps)
    "clops" => Some(Stages::CloPS)
    "ssacfg" => Some(Stages::SsaCfg)
    // NOTE: add stages here.
    "riscv" => Some(Stages::Asm)
    "finished" => Some(Stages::Finished)
    _ => None
  }
}

fn Stages::next(self : Stages) -> Stages {
  match self {
    Stages::Parse => Stages::Typecheck
    Stages::Typecheck => Stages::PreCps
    Stages::PreCps => Stages::Cps
    Stages::Cps => Stages::CloPS
    Stages::CloPS => if emit_js.val { Stages::Js } else { Stages::SsaCfg }
    Stages::Js => Stages::Finished
    Stages::SsaCfg => Stages::Asm
    // NOTE: add stages here.
    Stages::Asm => Stages::Finished
    Stages::Finished => Stages::Finished
  }
}

struct CompileStatus {
  mut curr_stage : Stages
  end_stage : Stages
  mut source_code : String?
  mut ast : @types.Syntax?
  mut typechecked : @types.Syntax?
  mut counter : Int // for unique var generation
  mut precps : @precps.PreCps?
  mut cps : @cps.Cps?
  mut clops : @closureps.ClosurePS?
  mut cfg : @ssacfg.SsaCfg?
  mut js_emitted : String
  // NOTE: add stages here.
  mut asm : Array[@riscv.AssemblyFunction]?
}

fn CompileStatus::initialize(
  start_stage : Stages,
  end_stage : Stages,
  file : String
) -> CompileStatus! {
  let v = CompileStatus::{
    curr_stage: start_stage,
    end_stage,
    source_code: None,
    ast: None,
    typechecked: None,
    counter: 0,
    precps: None,
    cps: None,
    clops: None,
    cfg: None,
    js_emitted: "",
    // NOTE: add stages here.
    asm: None,
  }
  match start_stage {
    Parse => v.source_code = Some(file)
    Typecheck => v.ast = Some(@types.Syntax::from_json!(@json.parse!(file)))
    PreCps =>
      v.typechecked = Some(@types.Syntax::from_json!(@json.parse!(file)))
    stage => @util.die("invalid start stage \{stage}")
  }
  v
}

/// Run the next stage of compilation. Returns true if the compilation is finished.
fn CompileStatus::step(self : CompileStatus) -> Bool {
  if self.curr_stage >= self.end_stage {
    return true
  }
  match self.curr_stage {
    Parse => {
      let source_code = self.source_code.unwrap()
      let parsed = @parser.parse_program(source_code)
      self.ast = Some(parsed)
    }
    Typecheck => {
      let to_check = self.ast.unwrap().clone()
      self.typechecked = Some(@typing.infer_type(to_check))
    }
    PreCps => {
      let tyenv = @precps.TyEnv::new(@types.externals)
      let entry_inlined = @precps.inline_entry(self.typechecked.unwrap())
      self.precps = Some(tyenv.ast2precps(entry_inlined))
      self.counter = tyenv.counter.val
    }
    Cps => {
      let cpsenv = @cps.CpsEnv::new(self.counter)
      let mut cps = cpsenv.precps2cps(self.precps.unwrap(), fn { _ => Exit })
      cps = @cps.optimize_cps(cps)
      self.cps = Some(cps)
      self.counter = cpsenv.counter.val
    }
    CloPS => {
      let clops = @closureps.cps2clops(self.counter, self.cps.unwrap())
      self.clops = Some(clops)
      self.counter = clops.counter.val
    }
    SsaCfg => {
      let cfg = @ssacfg.clops2ssacfg(self.clops.unwrap())
      self.cfg = Some(cfg)
    }
    Js => {
      let emitter = @js.JsEmitter::new(self.clops.unwrap())
      self.js_emitted = emitter.emit()
    }
    // NOTE: add stages here.
    Asm => {
      let real_asm = @riscv.emit(self.cfg.unwrap())
      self.asm = Some(real_asm)
    }
    Finished => ()
  }
  self.curr_stage = self.curr_stage.next()
  self.curr_stage >= self.end_stage
}

fn CompileStatus::output(self : CompileStatus, json : Bool) -> String {
  if json {
    match self.curr_stage {
      Parse => self.source_code.unwrap()
      Typecheck => @json.stringify(self.ast.unwrap().to_json())
      Finished => @riscv.print_functions(self.asm.unwrap())
      stage => @util.die("Can't make json output of \{stage}")
    }
  } else {
    match self.curr_stage {
      Parse => self.source_code.unwrap()
      Typecheck => self.ast.unwrap().to_string()
      PreCps => self.typechecked.unwrap().to_string()
      Cps => self.precps.unwrap().to_string()
      CloPS => self.cps.unwrap().to_string()
      SsaCfg | Js => self.clops.unwrap().to_string()
      Asm => self.cfg.unwrap().to_string()
      // NOTE: add stages here.
      Finished =>
        if emit_js.val {
          self.js_emitted
        } else {
          @riscv.print_functions(self.asm.unwrap())
        }
    }
  }
}

fn main {
  let argv = @env.get_args()
  let mut file = None

  // Testing directives
  let json = Ref::new(false)
  let start_stage = Ref::new(Stages::Parse)
  let end_stage = Ref::new(Stages::Finished)
  let closureps_interpreter = Ref::new(false)
  let out_file = Ref::new("-")
  let print = Ref::new([])
  @ArgParser.parse(
    [
      (
        "--start-stage",
        "",
        @ArgParser.String(
          fn(s) {
            match Stages::from_string(s) {
              Some(stage) => start_stage.val = stage
              None => @util.die("Invalid stage")
            }
          },
        ),
        "Start stage",
      ),
      (
        "--end-stage",
        "",
        @ArgParser.String(
          fn(s) {
            match Stages::from_string(s) {
              Some(stage) => end_stage.val = stage
              None => @util.die("Invalid stage")
            }
          },
        ),
        "End stage",
      ),
      ("--js", "", @ArgParser.Set(emit_js), "Emits js"),
      ("--wasm", "", @ArgParser.Set(emit_wasm), "Emits wasm"),
      (
        "--clops-interp",
        "",
        @ArgParser.Set(closureps_interpreter),
        "Run with closure passing style interpreter",
      ),
      (
        "--out-file",
        "-o",
        @ArgParser.String(fn(s) { out_file.val = s }),
        "Output file",
      ),
      ("--json", "", @ArgParser.Set(json), "Print results as JSON"),

      // Regular directives
      (
        "--print",
        "",
        @ArgParser.String(fn(s) { print.val = s.split(",").collect() }),
        "",
      ),
    ],
    fn(s) {
      if file.is_empty().not() {
        @util.die("multiple files are given")
      }
      file = Some(s)
    },
    "",
    argv,
  )

  // Configure pipeline
  let stages_to_print = print.val.map(
    fn(s) {
      match Stages::from_string(s) {
        Some(stage) => stage
        None => @util.die("Invalid stage to print: \{s}")
      }
    },
  )

  // Input
  let file = if file.is_empty() {
    @util.die("no input file provided")
  } else {
    file.unwrap()
  }
  let contents = @fs.read_to_string(file)
  //@util.die(contents)

  // Compilation
  let status = match
    CompileStatus::initialize?(start_stage.val, end_stage.val.next(), contents) {
    Ok(status) => status
    Err(e) => {
      println(e)
      @util.die("Initialization error")
    }
  }

  // Run stages
  while true {
    let stg = status.curr_stage
    let stop = status.step()
    if stages_to_print.contains(stg) {
      println("Stage: \{stg}")
      println(status.output(json.val))
    }
    if stop {
      break
    }
  }

  // Output
  if closureps_interpreter.val {
    let clops = status.clops.unwrap()
    let interpreter = @closureps_eval.CloPSInterpreter::new(clops)
    add_clops_interp_fns(interpreter)
    try {
      interpreter.eval!(clops.root)
    } catch {
      VariableNotFind(v) => println("Undefined variable: \{v}")
    } else {
      v => ignore(v)
    }
  } else {
    let out_string = status.output(json.val)
    if out_file.val == "-" {
      println(out_string)
    } else {
      @fs.write_to_string(out_file.val, out_string)
    }
  }
}
